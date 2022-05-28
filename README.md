# PVCC
Parsing, Vectorizing, Clustering and Classifying of articles from the website
# Кратко
Источник статей: https://sapr.ru/<br/>
Спарсено ~4к статей.<br/>
Данные статьи векторизованы с помощью библиотеки W2V - для каждой статьи получены вектора. Сжатие было осуществлено с помощью PCA и UMAP. Второй метод показал лучшую кластеризацию на данном наборе данных.<br/>
<p align="center">
  <img src="сжатие_PCA.png" width="400">
  <img src="сжатие_UMAP.png" width="400">
</p>
Была произведена кластеризация по силуэту выборки и по метрике Дэвиса–Булдина.<br/>
<p align="center">
  <img src="оценка_общая.png"">
</p>
Затем был обучен классификатор. В качестве классификатора использовался метод ближайших соседей из библиотеки Scikit-learn ввиду довольно легкой различимости классов<br/>
<div>
Итоговые результаты:
<ul>
  <li>Итоговая доля правильных ответов: 100.00%</li>
  <li>Итоговая точность: 100.00%</li>
  <li>Итоговая полнота: 100.00%</li>
  <li>Итоговая метрика f1: 100.00%</li>
</ul>
Стопроцентный результат получен из-за наличия четкой, широкой границы между классами и отсутствия выбросов 🙃
</div>



