# Analyse exploratoire des données (EDA)
Analyse approfondie du jeu de données Movies et Users pour comprendre les tendances d'évaluation, le comportement des utilisateurs et les caractéristiques des films.

## Distribution des ratings:
La distribution des notes **n’est pas uniforme** et montre une forte concentration autour des valeurs élevées, en particulier la note 4, tandis que la majorité des utilisateurs attribuent des notes comprises entre 3 et 5 et utilisent très rarement les notes basses (0.5 et 1.5).

<p align="center" style="margin: 40px 0;">
  <img src="../figures/distribution_ratings.png" alt="Distribution des ratings" width="420" height="420" />
</p>

Cette **asymétrie** traduit un biais positif global et des biais individuels de notation, les utilisateurs ayant tendance à noter surtout les films qu’ils ont appréciés et à utiliser seulement une partie de l’échelle disponible. Dans ce contexte, les modèles basés sur des similarités directes (comme le KNN naïf) risquent d’être peu discriminants et sensibles au bruit, alors qu’un modèle de Matrix Factorization intégrant des biais utilisateurs et items (par exemple SVD ou BigQuery ML MATRIX_FACTORIZATION) est plus adapté pour capter ces biais, lisser le bruit et exploiter efficacement l’information contenue dans les ratings.


## Nombre de films par genre:
Les films ont de nombreux genres bien répartis, l’information de contenu est **assez  discriminante** et peut justifier un modèle hybride combinant factorisation matricielle et features de contenu

<p align="center" style="margin: 40px 0;">
  <img src="../figures/genres_per_movie.png" alt="Genre par movie" width="420" height="420" />
</p>

## Films à longue queue :
Ce graphique met en évidence une distribution en longue traîne très marquée de la popularité des films. On observe clairement que la **grande majorité des films n’ont reçu qu’un très petit nombre de ratings** (quelques unités à une dizaine), tandis qu’un nombre très limité de films concentre un volume élevé de notes (plusieurs dizaines voire centaines). 


<p align="center" style="margin: 40px 0;">
  <img src="../figures/long_tail_movies.png" alt="Long tail movies" width="420" height="420" />
</p>

Les modèles basés sur des similarités explicites entre films (comme le KNN item-based) deviennent peu fiables, car ils manquent de données suffisantes pour calculer des distances pertinentes pour la majorité des films. 
De même, les approches basées uniquement sur la popularité favorisent excessivement les films déjà très connus et ignorent la richesse de la longue traîne. À l’inverse, cette configuration est idéale pour un modèle de Matrix Factorization

## Biais des films:

Ce graphique révèle un biais de popularité marqué : les films avec **peu de ratings (< 10)**  présentent une **forte variance** de notes (0.5 à 5.0) tandis que les films populaires convergent vers des moyennes stables (3.0-4.5). 

<p align="center" style="margin: 40px 0;">
  <img src="../figures/movie_bias.png" alt="Movie bias" width="420" height="420" />
</p>


Pour ce contexte,  les modèles les plus efficaces sont les modèles avec régularisation comme la factorisation matricielle (SVD, ALS) ou des approches hybrides combinant filtrage collaboratif et contenu, car ils gèrent mieux les données éparses et le problème de "cold start".Il y'a de fortes chances que les modèles simples (K-NN non pondéré, moyenne simple) deviennent trop sensibles aux films peu notés.

## Ratings par film:
Dans ce graphique on remarque que la majorité des films ont **moins de 20 notes**.  Cela renforce encore plus la nécessité d'utiliser des modèles robustes à la rareté des données : il faut privilégiez absolument la factorisation matricielle avec forte régularisation (SVD++, ALS avec paramètres de régularisation élevés) ou des modèles hybrides qui exploitent les métadonnées des films (genre, réalisateur, acteurs) pour compenser le manque de ratings. 

<p align="center" style="margin: 40px 0;">
  <img src="../figures/ratings_per_movie.png" alt="Ratings par movie" width="420" height="420" />
</p>


Les approches basées uniquement sur le filtrage collaboratif pur seront très limitées car elles n'auront pas assez de signal pour la majorité des films.

## Ratings par utilisateur:
Un autre graphique montre que la majorité des utilisateurs  des ont notés **moins de 200 films**. Les modèles basés sur la similarité directe (User-User CF) seront peu efficaces. 

<p align="center" style="margin: 40px 0;">
  <img src="../figures/ratings_per_user.png" alt="Genre par movie" width="420" height="420" />
</p>


Il vaut mieux privilégier ALS / Matrix Factorization ou modèles hybrides qui intègrent des embeddings et des features supplémentaires pour gérer le cold-start et la sparsité.

## Activité de l'utilisateur:

Ce graphique montre une distribution très déséquilibrée de l'activité utilisateurs : la majorité des utilisateurs ont noté relativement **peu de films (10-100 ratings)**, avec une concentration importante autour de 20-50 ratings par utilisateur, tandis qu'une minorité d'utilisateurs très actifs ont noté plusieurs centaines voire milliers de films. 

<p align="center" style="margin: 40px 0;">
  <img src="../figures/user_activity.png" alt="User activity" width="420" height="420" />
</p>


Comme on a dit précédemment il vaut mieux privilégier ALS / Matrix Factorization ou modèles hybrides qui intègrent des embeddings et des features supplémentaires pour gérer le cold-start et la sparsité. 

## Bias des utilisateurs:
Ce graphique révèle des biais utilisateurs hétérogènes cruciaux pour notre système de recommandation : l'écart-type (variance) des notes varie considérablement selon les utilisateurs, avec une **concentration autour de 0.75-1.25**, indiquant que certains utilisateurs notent de manière très cohérente (faible variance) tandis que d'autres sont très volatils dans leurs évaluations. 

<p align="center" style="margin: 40px 0;">
  <img src="../figures/user_bias.png" alt="Genre par movie" width="420" height="420" />
</p>

On observe aussi que les utilisateurs ayant une moyenne de ratings élevée (4.0-5.0) tendent à avoir une variance plus faible. Pour notre modèle, cela implique la nécessité absolue d'une normalisation par utilisateur.