# Пример №2

Для проекта использована сеть *keypointrcnn_resnet50_fpn*, на данном примере показана ее работа с тестовым видео, на котором танцуют 2 танцора. Тот, что слева - "коуч", тот что справа - "обучаемый". Проводится анализ исходного видео, детектирование танцующих, выделение ключевых точек, нанесение скелета. Вычесленные метрики воспроизводятся на итоговом видео.

В данном ноутбуке подробно расписан процесс поиска решения, проводятся эксперименты.

В данном примере была решена проблема путаницы нейросетью "коуча" и "обучаемого" (см. *example_1*).
     
## Структура

*/frames* - папка с фреймами из начального видео       
*/frames_output* - папака с обработанными фреймам (нанесены ключевые точки и скелет с метриками)       
*/input_video* - папка, где расположено начальное видео для анализа        
*/output_video* - папка, где расположено итоговое видео (>25Мб, поэтому добавлено в .gitignore)        
         
Воспроизводимость:        
*project.yaml* - рабочее окружение Anaconda  

**Примечание:**
- для облегчения репозитрия в папках с фреймами оставлены только по одному фрейму 
- т.к. выходной видеофайл весит > 25Мб он добавлен в *.gitignore*
   
## Примеры

Пример фрейма (кадра) до обработки:      
<image src="frames/frame_10.jpg" width="600">

Пример фрейма (кадра) после обработки:       
<image src="frames_output/frame_10.jpg"  width="600">