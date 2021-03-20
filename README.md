# LaTeX Papers Template

Шаблонный репозиторий для написания документов в LaTeX.

## Настройка репозитория

1. Форкните репозиторий к себе в удобное место.
1. Склонируйте его на локальную машину.
1. Выполните скрипт `setup.sh`.
1. Установите *Visual Studio Code* и расширение *LaTeX Workshop*.
1. Вызовите команду `lpt update` для проверки работы утилиты `lpt`.

## Создание документа

1. Выберите некоторый шаблон документа из папки `templates`. 
1. Вызовите команду `lpt create template_name paper_name`. 
1. Удалите все файлы `.gitkeep`.
1. В корневом latex-файле правильно задайте значение переменной `\pwd` как относительный путь до папки с документом от корня репозитория.
1. Запустите сборку документа из интерфейса VSCode: вкладка *TeX* в левой боковой панели, далее раскрыть *Build LaTeX project*, далее *Recipe: using docker*. Если данный *Recipe* не отображается в списке, то следует перезагрузить VSCode.
1. Альтернативным вариантом сборки может быть клик по кнопке *Build LaTeX project*, показывающуюся в верхнем меню справа при открытии корневого latex-файла.
1. Артефакты сборки должны располагаться в папке `out` внутри папки с документом. 

## Дополнительно

1. Регулярно обновляйте репозиторий при помощи команды `lpt update`.
1. Просмотрите содержимое файла `.protection`. Не вносите изменения в файлы, расположенные по перечисленным в этом файле путям, так как это может повлечь за собой merge conflicts при попытке обновления репозитория.
1. В файле `.vscode/snippets.code-snippets` записаны сниппеты, в которых заданы шаблоны вставки изображений, таблиц, листингов и прочего. Рекомендуется использовать именно эти сниппеты при работе над документами, так как они гарантируют корректность оформления документа.
