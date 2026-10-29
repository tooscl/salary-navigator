"""Execute the actual batch JavaScript with in-memory browser/filesystem doubles."""
import json
from pathlib import Path
import shutil
import subprocess
import unittest


class OptionalFieldsBatchTests(unittest.TestCase):
    def execute_batch(self, invalid=None):
        if not shutil.which('node'):
            self.skipTest('Node is needed for the JS regression')
        batch = (Path(__file__).parent/'skills/collect-avito/scripts/browser_batch.js').read_text()
        job = {'mode':'capture', 'completed_ids':[], 'queue':[
            {'id':'123','url':'123','listing_rank':1},
            {'id':'456','url':'456','listing_rank':2},
            {'id':'789','url':'789','listing_rank':3}]}
        batch = batch.replace('__JOB__', json.dumps(job)).replace('__EXTRACTOR__', 'null')
        prefix = '''
const files = new Map(); let current = null; const opened = [];
const fs = {mkdir:async()=>{}, writeFile:async(name, data)=>files.set(name,data),
 rename:async(from,to)=>{files.set(to, files.get(from));files.delete(from);},
 readFile:async(name)=>files.get(name)};
const pwd = '/memory';
const p = {url:()=>current, goto:async(id)=>{current=id;opened.push(id);}, evaluate:async()=>
 current==='456' ? {state:'NEEDS_HUMAN', id:'456'} :
 {state:'DETAIL',id:current,title:'Курьер',description:null,location_raw:[],
 parameter_audit:{rendered_complete:true},fields:[{name:'График',value:'Гибкий'}], ...INVALID}};
const listBrowserTabs = async()=>current ? [{targetId:'test-tab',url:current}] : [];
const openTab = async(id)=>{await p.goto(id);return p;};
const snapshot = async()=>({tree:'fixture'}); const closeTab = async()=>{};
(async()=>{
'''
        suffix = '''
console.log('RESULT='+JSON.stringify({opened, checkpoint:JSON.parse(files.get('checkpoint.json')),
 raw:JSON.parse(files.get('raw/123.json'))}));
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
        prefix = prefix.replace('INVALID', json.dumps(invalid or {}, ensure_ascii=False))
        result = subprocess.run([shutil.which('node'), '-'], input=prefix+batch+suffix,
                                text=True, capture_output=True, check=True)
        value = json.loads(next(line.removeprefix('RESULT=') for line in result.stdout.splitlines()
                                if line.startswith('RESULT=')))
        return value

    def test_optional_gaps_are_saved_and_access_block_still_stops(self):
        value = self.execute_batch()
        self.assertEqual(value['opened'], ['123','456'])
        self.assertEqual(value['checkpoint']['completed_ids'], ['123'])
        self.assertEqual(value['checkpoint']['state'], 'NEEDS_HUMAN')
        self.assertIsNone(value['raw']['description'])
        self.assertEqual(value['raw']['location_raw'], [])

    def test_malformed_optional_fields_stop_before_next_card(self):
        for invalid in [{'description':123}, {'fields':None}, {'fields':[{'name':'График','value':12}]}]:
            with self.subTest(invalid=invalid):
                value = self.execute_batch(invalid)
                self.assertEqual(value['opened'], ['123'])
                self.assertEqual(value['checkpoint']['completed_ids'], [])
                self.assertEqual(value['checkpoint']['state'], 'REVIEW_REQUIRED')
