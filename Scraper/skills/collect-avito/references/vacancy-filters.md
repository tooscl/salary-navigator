# Проверенный путь: профессия и город внутри вакансий

27.10.2026 воспроизведён Terra, затем отдельно **Luna low без ошибок** на
странице пользователя `/moskva/vakansii?context=…`. Обе проверки подтвердили
Екатеринбург, отмеченную профессию, пустой поиск; карточки не открывались.
Сначала выбирается профессия, затем наблюдённая ссылка города, сохраняющая
раздел и профессию. Настройка фильтров не требует главной или общего каталога городов. Для
технически недоступного начального входа отдельно предусмотрен
[запасной маршрут версии 1.4](vacancy-entry-routes.md).

## Карта наблюдённых элементов

```xml
<page path="/moskva/vakansii">
  <form>
    <profession>
      <label role="checkbox" data-marker="params[144170]/checkbox/22283175"
             aria-checked="false|true">
        <input name="Водитель грузового транспорта" value="22283175" />
      </label>
    </profession>
  </form>
  <a title="Екатеринбург" href="/ekaterinburg/vakansii/…">Екатеринбург</a>
  <footer>
    <a data-marker="footer/regions/seo-link" href="/ekaterinburg">Екатеринбург</a>
  </footer>
</page>
```

Это компактная карта фактически наблюдённых узлов, не полный DOM и не гарантия
неизменности интерфейса. Номера refs `e…` динамические: брать из свежего snapshot.
Ссылку из `footer/regions/seo-link` не выбирать: она теряет контекст вакансий.
`avito.st` не является страницей выдачи или маршрутом выбора города.

## Готовые действия для Luna

Сначала выберите подключённый профиль по разделу SKILL.md «Подключённый профиль
Aside». Используйте **один интерактивный**
`aside --host <host> --profile <ID> repl` (при необходимости `--account <account>`);
те же TL_ASIDE_HOST/TL_ASIDE_ACCOUNT/TL_ASIDE_PROFILE передавайте runner.
JS передавайте в stdin через
`write_stdin(chars=...)`. Не вкладывайте CSS-селекторы в shell-строки: это уже
вызывало ошибку quoting. Актуальные правила `/browse` остаются обязательными.
Для каждого отдельного вызова exec явно задавайте `workdir` подготовленного
checkout: рабочий каталог не сохраняется между вызовами. Вызывайте `run.py`
по абсолютному пути внутри этого checkout. 29.10.2026 запуск capture из основной
копии был остановлен на preflight до чтения карточек; продолжение разрешено из
выделенного checkout с сохранённым checkpoint и журналом ошибок.
В Aside REPL `pwd` — значение, а не функция: используйте `pwd`, не `pwd()`.
Это подтверждено `aside guide repl`; 29.10.2026 вызов `pwd()` дал TypeError
при сохранении уже полученного доказательства пагинации.

1. `var p = await openTab(startURL)` — точная ссылка пользователя на `/vakansii`.
2. `console.log((await snapshot(p,{selector:'form',interactive:true})).tree)`.
3. После подтверждения узла флажка профессии в snapshot:

   Если свежий snapshot показывает вложенный checkbox, используйте его точный
   ref через Aside: `await p.locator('<ref из snapshot>').check()`. Каждый новый
   snapshot инвалидирует старые refs; не угадывайте ref. Символ `x` в дереве не
   добавляется к имени для role selector. После действия снимите новый snapshot
   и проверьте `[checked]`. 29.10.2026 на «Комплектовщик» в Москве Luna
   подтвердил `locator('e183').check()` и checked=true; ref здесь только пример.
   Для наблюдённого label с marker ранее проверен и такой способ:

   ```js
   var professionControl = p.locator('label[role="checkbox"][data-marker="params[144170]/checkbox/22283175"]');
   if (await professionControl.getAttribute('aria-checked') !== 'true') {
     await professionControl.click();
   }
   console.log((await snapshot(p,{selector:'form',interactive:true})).tree);
   ```

   Флажок должен быть `[checked]`. Поле «Должность» — текстовый поиск, его не
   заполнять вместо фильтра профессии. Если список профессий скрыт, использовать
   его фактический disclosure из snapshot; не переключаться на популярный тег.
   Для нескольких профессий сохраните тот же рабочий метод: открыть наблюдённое
   «Показать ещё» группы, искать название только через
   `params[144170]/multiselect-search/input/input`, выбирать наблюдённый точный
   label с `params[144170]/checkbox/<site_value>`. Глобальный поиск объявлений
   для этого не использовать. При наблюдённом обновлении панели допустимо
   ожидание до 5 секунд. После успешного выбора списка не запускайте дополнительную
   ревизию всех отметок, не требуйте их наличия в свёрнутой панели.
   Для наших топ-20 фиксируйте выполненный выбор через --profession-scope top20,
   для согласованной выдачи без фильтра — --profession-scope all. Дальше runner
   использует наблюдённую Next; повторных проверок профессий и сортировки нет.
4. Прочитать реальные городские ссылки на обновлённой профессиональной выдаче:

   ```js
   var cityLinks = await p.evaluate(() => Array.from(document.querySelectorAll('a[title="Екатеринбург"]'))
     .map(a => ({title:a.title, href:a.getAttribute('href'), marker:a.getAttribute('data-marker')})));
   console.log(cityLinks);
   ```

   Выбрать наблюдённую ссылку с `/ekaterinburg/vakansii/` и той же профессией.
   В успешной проверке href был
   `/ekaterinburg/vakansii/voditel_gruzovogo_transporta-ASgBAgICAUTUzBHOjqAV`.
   Это пример результата, не шаблон для конструирования URL.
5. Кликнуть именно по обнаруженному href:

   ```js
   // observedCityHref — точное значение из cityLinks, проверенное на шаге 4.
   await p.locator('a[title="Екатеринбург"][href='+JSON.stringify(observedCityHref)+']').click();
   console.log((await snapshot(p,{selector:'form',interactive:true})).tree);
   ```

6. Подтвердить одновременно:

   ```js
   console.log(await p.evaluate(() => {
     const input = Array.from(document.querySelectorAll('input')).find(e => e.value === '22283175');
     const search = document.querySelector('input[placeholder="Поиск по объявлениям"]');
     return {url:location.href, heading:document.querySelector('h1')?.innerText,
       profession:input?.name, checked:input?.checked,
       aria_checked:input?.closest('label')?.getAttribute('aria-checked'), query:search?.value};
   }));
   ```

   Проверенный результат: URL в `/ekaterinburg/vakansii/`, heading
   «Вакансии по профессии Водитель грузового транспорта в Екатеринбурге»,
   checked=true, aria_checked=true, query="". Неизвестное query не равно пустому.
   Только после этой проверки передавать URL в `run.py freeze`.

Для другой профессии/города наблюдайте их фактические name/value/href, не
подставляйте неизвестные ID. Если нужного маршрута нет в паспорте, сначала
используйте географический контроль внутри вакансий «Город, район, метро» /
«Москва, район, метро» (наблюдённый marker `search-form/change-location`).
Путь: «Город или регион» → фактическая подсказка → «Показать …» → подтверждение
выдачи и всех фильтров. Неизвестный город или отсутствие подсказки не означает
успешный выбор. Запасной путь через главную разрешён после неудачи основного;
порядок, запись новых маршрутов и статус «Всех регионов» описаны в
[паспорте маршрутов](vacancy-entry-routes.md).

## Ошибки и новые действия

Первую ошибку API, quoting, отсутствующий элемент или неприменённый фильтр
сразу записать через `run.py record-error --stage filters`. Затем разрешена
до пяти повторов с объяснением конкретной коррекции перед каждым, через
`check-stage --stage filters --retry-reason '…'`. Успех подтвердить фактическим
snapshot и `stage-ok --stage filters --evidence '…'`. Счётчик ошибок не сбрасывать.
Дубли check-stage одного активного этапа не тратят повтор. После исчерпания
пяти повторов в любом этапе run_id — NEEDS_HUMAN и краткий отчёт человеку.
CAPTCHA/429/ограничение/неверный ID останавливают немедленно. Передача Terra
не даёт новых попыток. Новое разрешение человека — resume-browser по SKILL.md.

Аудит 29.10.2026: role selector с именем `xКомплектовщик` не нашёл элемент;
`check-stage --immediate` отклонён argparse: этот флаг поддерживает только
`record-error`. Обе ошибки сохранены в локальном журнале. После разрешения с ID
`standing-18-f04-cli-correction-20261029` сделана одна коррекция по свежему ref,
подтверждённая snapshot. Штатный resume открывает бюджет runner; более строгое
ограничение участника на число действий обязательно соблюдать отдельно.

Если понадобилось написать новый JS/Python-скрипт или добавить действие,
сохранить его как переиспользуемый файл проекта, описать точный вызов здесь
или в SKILL.md, добавить проверку и включить всё в тот же PR. Не оставлять
единственный работающий вариант в /tmp или истории чата. Неизменённые уже
проверенные шаги повторно не переписывать.
