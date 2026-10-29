// Invoked by run.py through gstack /browse's Aside REPL, never a separate browser.
const job = __JOB__;
const extractor = __EXTRACTOR__;
console.log('TL_SANDBOX=' + pwd);
await fs.mkdir('raw', {recursive: true});
let p = globalThis.tlCollectorPage || null;
let external = Boolean(job.tab_id);
let targetId = job.tab_id || null;
const safeUrl = value => { try { const u = new URL(value); return u.origin + u.pathname; } catch (_) { return null; } };
const saveTab = async () => save('browser-tab.json', {target_id: targetId, ownership: external ? 'external' : 'collector', url: p ? safeUrl(p.url()) : safeUrl(job.url || job.queue?.[0]?.url)});
const done = [...job.completed_ids];
const save = async (name, data) => {
  await fs.writeFile(name + '.tmp', JSON.stringify(data));
  await fs.rename(name + '.tmp', name);
};
const checkpoint = async (state, next, reason = null) => save('checkpoint.json',
  {state, completed_ids: done, next_id: next, stop_reason: reason});
const started = Date.now();
try {
  if (!p && targetId) p = await attachBrowserTab(targetId);
  if (job.mode === 'freeze') {
    if (!p) {
      const before = new Set((await listBrowserTabs()).map(t => t.targetId));
      p = await openTab(job.url);
      const added = (await listBrowserTabs()).filter(t => !before.has(t.targetId) && t.url === p.url());
      if (added.length === 1) targetId = added[0].targetId;
    } else if (job.ui_next && !job.previous) {
      await p.goto(job.url);
    } else if (!job.ui_next && job.allow_navigation && p.url() !== job.url) await p.goto(job.url);
    await saveTab();
    const stable = async (accept = () => true) => {
      let last = null, repeats = 0;
      for (let i = 0; i < 30; i++) {
        const value = await p.evaluate(extractor);
        if (value.state === 'NEEDS_HUMAN') { await save('listing.json', value); throw new Error('NEEDS_HUMAN: access restriction'); }
        const {observed_at, ...stableFields} = value;
        const key = JSON.stringify(stableFields);
        repeats = key === last ? repeats + 1 : 0; last = key;
        if (repeats >= 2 && value.state === 'LISTING' && value.items.length && value.listing_item_audit?.malformed === 0 && accept(value)) return value;
        await sleep(500);
      }
      throw new Error('Listing did not stabilize');
    };
    let listing;
    if (job.ui_next && job.previous) {
      const previous = job.previous, first = job.first;
      const pageNo = u => { const n = new URL(u).searchParams.getAll('p'); if (n.length > 1 || (n.length && !/^[1-9]\d*$/.test(n[0]))) throw new Error('Invalid page number'); return Number(n[0] || 1); };
      const checked = l => JSON.stringify((l.selected_controls || []).filter(c => (c.marker || '').startsWith('params[144170]/checkbox/')).map(c => [c.marker, c.value, c.label]).sort());
      const valid = (l, n) => l.state === 'LISTING' && new URL(l.source_url).pathname === new URL(first.source_url).pathname && pageNo(l.source_url) === n && (l.pagination?.current_pages || []).includes(String(n)) && l.heading === first.heading && checked(l) === checked(first);
      let current = await stable();
      const n = pageNo(previous.source_url);
      let recovered = true, observedHref = previous.pagination?.evidence?.href || null, nextSnapshot = null;
      if (valid(current, n) && current.source_url === previous.source_url) {
        recovered = false;
        const snapBefore = await snapshot(p, {selector: '[aria-label="Следующая страница"]', interactive: true});
        const line = snapBefore.tree.split('\n').find(x => /link "Следующая страница"/.test(x));
        nextSnapshot = line || null;
        const ref = line && line.match(/\[ref=(e\d+)\]/);
        observedHref = current.pagination?.evidence?.href || null;
        const resolvedHref = observedHref ? new URL(observedHref, current.source_url).href : null;
        const comparison = {from_url: current.source_url, expected_next: previous.pagination?.next_url || null,
          current_next: current.pagination?.next_url || null, observed_href: observedHref,
          resolved_href: resolvedHref, ref_found: Boolean(ref), scoped_snapshot: snapBefore.tree};
        await save('next-comparison.json', comparison);
        if (!ref || !observedHref || resolvedHref !== previous.pagination?.next_url || current.pagination?.next_url !== previous.pagination?.next_url)
          throw new Error('Observed Next differs from saved page; see next-comparison.json');
        await p.locator(ref[1]).click();
        current = await stable(l => valid(l, n + 1));
      }
      if (!valid(current, n + 1)) throw new Error('Tab is not the expected completed next page');
      listing = {...current, ui_transition: {method: 'observed_ui_next', recovered, next_snapshot: nextSnapshot, from_url: previous.source_url, from_page: n, next_href: previous.pagination.next_url, observed_href: observedHref, landed_url: current.source_url, landed_page: n + 1, stability: 'three_matching_snapshots_500ms'}};
    } else listing = job.ui_next ? await stable(l => (l.pagination?.current_pages || []).includes('1')) : await p.evaluate(extractor);
    const snap = await snapshot(p, {interactive: true});
    await save('listing.json', listing);
    if (listing.state !== 'LISTING') {
      console.log(snap.tree);
      await checkpoint(listing.state === 'NEEDS_HUMAN' ? 'NEEDS_HUMAN' : 'REVIEW_REQUIRED', null, listing.state);
    } else {
      await checkpoint('READY', null);
      console.log('TL_LISTING=' + JSON.stringify({heading: listing.heading,
        controls: listing.selected_controls, items: listing.items.length}));
    }
  } else {
    for (const item of job.queue) {
      await checkpoint('CAPTURING', item.id);
      if (!p) {
        const before = new Set((await listBrowserTabs()).map(t => t.targetId));
        p = await openTab(item.url);
        const added = (await listBrowserTabs()).filter(t => !before.has(t.targetId) && t.url === p.url());
        if (added.length === 1) targetId = added[0].targetId;
      } else await p.goto(item.url);
      await saveTab();
      const snap = await snapshot(p, {interactive: true});
      const data = await p.evaluate(extractor);
      // Preserve the observation before deciding whether it can enter the dataset.
      await save('raw/' + item.id + '.json', data);
      const validFields = Array.isArray(data.fields) && data.fields.length > 0 &&
        data.fields.every(f => f && typeof f.name === 'string' && f.name.trim() &&
          typeof f.value === 'string' && f.value.trim());
      const validDescription = data.description == null || typeof data.description === 'string';
      const ready = data.state === 'DETAIL' && data.id === item.id &&
        safeUrl(p.url()) === safeUrl(item.url) && safeUrl(data.source_url) === safeUrl(item.url) &&
        data.parameter_audit?.rendered_complete && typeof data.title === 'string' &&
        data.title.trim() && validFields && validDescription;
      if (!ready) {
        await fs.writeFile('problem.snapshot.txt', snap.tree);
        console.log(snap.tree);
        await checkpoint(data.state === 'NEEDS_HUMAN' ? 'NEEDS_HUMAN' : 'REVIEW_REQUIRED',
          item.id, data.reason || 'identity_or_template_review');
        break;
      }
      done.push(item.id);
      await checkpoint('CAPTURED', null);
      console.log('TL_CARD=' + JSON.stringify({id: item.id, rank: item.listing_rank,
        fields: data.fields.length, description_characters: (data.description || '').length}));
    }
  }
} catch (error) {
  await checkpoint(String(error).includes('NEEDS_HUMAN:') ? 'NEEDS_HUMAN' : 'BROWSER_ERROR', job.queue?.find(i => !done.includes(i.id))?.id || null, String(error));
  console.log('TL_ERROR=' + String(error));
} finally {
  await saveTab();
  await save('timing.json', {capture_seconds: (Date.now() - started) / 1000});
  const cp = JSON.parse(await fs.readFile('checkpoint.json', 'utf8'));
  if (p && !external && job.close_tab && (cp.state === 'READY' || cp.state === 'CAPTURED')) {
    await closeTab(p); p = null;
  }
  globalThis.tlCollectorPage = p;
  globalThis.tlCollectorExternal = external;
  console.log('TL_FINISHED');
}
