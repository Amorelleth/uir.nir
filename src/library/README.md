# Оформление библиографии

## Рекомендации

1. В файле `examples.bib` приведены примеры различных видов bibtex-записей.
1. Обязательно для каждой записи добавлять поле `language = {russian}`, потому что это правило заставляет отображать ссылку на источник литературы по ГОСТ.
1. Для удобства можно ввести иерархическую систему обозначений псевдонимов источников литературы. Например, `Book:DB:Third-Manifest`.
1. Для удобства можно аннотировать источники литературы, например следующим образом:
    ```tex
    % About: The basic work on CSP process model
    @online{online,
        author   = {Hoare C.A.R.},
        title    = {Communicating Sequential Processes},
        year     = {2015},
        url      = {http://www.usingcsp.com/cspbook.pdf},
        urldate  = {18.12.20},
        language = {russian}
    }
    ```
1. VSCode поддерживает форматирование bib-файлов.
