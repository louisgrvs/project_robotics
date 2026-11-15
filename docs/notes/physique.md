# Paramètres physiques du robot simulé

Jour 3 du programme. Tous les chiffres ci-dessous sont produits par
`scripts/validate_kinematics.py` (résultats dans `results/validate_kinematics.json`,
figure dans `docs/figures/glissement_mu.png`). Commit de référence : *à compléter*.

## Paramètres retenus

| Paramètre | Valeur | Où |
|---|---|---|
| Rayon des roues r | 0,05 m | `robot.xml` |
| Entraxe L | 0,24 m (géométrique = effectif) | `robot.xml` |
| Forme de contact des roues | disque ellipsoïdal, `size="0.05 0.05 0.01"` | `robot.xml` |
| Origine du châssis | milieu de l'essieu (`base_link`, REP 105) | `robot.xml` |
| Vitesse maximale d'une roue | 10 rad/s (`ctrlrange`), soit 0,5 m/s | `robot.xml` |
| Couple maximal d'une roue | 0,2 N·m (`forcerange`) | `robot.xml` |
| Gain de l'actionneur k<sub>v</sub> | 1 N·m·s/rad | `robot.xml` |
| Frottement roues–sol μ | 1,0 (nominal) | `robot.xml` |
| Roulette | sans frottement (`condim="1"`, `priority="1"`) | `robot.xml` |
| Intégrateur | `implicitfast` | `robot.xml` |
| Cône de frottement | `elliptic` | `robot.xml` |
| Pas de temps Δt | 2 ms (500 Hz, soit 50 pas par décision à 10 Hz) | défaut MuJoCo |
| Sol | plan à z = 0, paramètres de contact par défaut | `arena.xml` |

Masse totale : 1,26 kg. Au repos, chaque roue porte 4,49 N et la roulette le reste
(27 % du poids, 12,4 N).

## Résultats de validation

Mesures au milieu de l'essieu, en régime établi (de 2 à 4 s après la commande).
Les colonnes « actionneur » et « cinématique » décomposent l'écart total : la
première compare la vitesse des roues à la consigne, la seconde compare le mouvement
du robot à ce que prédisent les vitesses de roue mesurées.

| Manœuvre (ω<sub>L</sub>, ω<sub>R</sub>) | Grandeur | Théorie | Mesure | Écart | Actionneur | Cinématique | Seuil |
|---|---|---|---|---|---|---|---|
| Ligne droite (5, 5) | v | 0,2500 m/s | 0,2492 m/s | 0,34 % | 0,00 % | 0,34 % | 2 % |
| Rotation sur place (−5, 5) | ω | 2,0833 rad/s | 2,0764 rad/s | 0,33 % | 0,00 % | 0,33 % | 3 % |
| Cercle (3, 6) | R | 0,3600 m | 0,3592 m | 0,21 % | 0,13 % | 0,09 % | 3 % |

Quand Δt passe de 2 ms à 1 ms, les trois mesures varient de moins de 0,001 %
(seuil : 1 %). Elles sont identiques de 0,5 à 8 ms avec `implicitfast`.

## Justification des choix

### Forme des roues : un ellipsoïde plutôt qu'un cylindre

Avec les roues cylindriques du jour 2, la rotation sur place échouait : 2,0013 rad/s
mesurés pour 2,0833 attendus, soit 3,94 % d'écart, dont 3,59 % de part cinématique.
Un cylindre posé sur un plan produit deux points de contact, aux deux bords de la
roue, espacés de 2 cm. En rotation, ces deux bords voudraient rouler à des vitesses
différentes. Ils glissent donc en sens opposés, avec une force tangentielle égale à
μ·F<sub>n</sub> sur chacun des quatre points.

L'entraxe effectif n'est alors plus défini par la géométrie mais par l'équilibre
de ces frottements : il se situe entre 0,22 et 0,26 m, et la mesure donne 0,25 m.
Trois indices le confirment. L'écart diminue quand μ baisse (0,25 % à μ = 0,2), car
le frottement parasite est plus faible. Il dépend du pas de temps (variation de
0,96 %, juste sous le seuil). Enfin, la ligne droite, où les deux bords roulent à la
même vitesse, n'est pas affectée (0,03 %).

C'est le phénomène que Borenstein et Feng (1996) décrivent comme l'incertitude sur
l'entraxe effectif d'un robot réel à roues larges. Un ellipsoïde aplati (rayon
5 cm, demi-épaisseur 1 cm) donne un seul point de contact : l'entraxe effectif
redevient l'entraxe géométrique, et l'écart en rotation tombe à 0,06 %.

Ce choix a deux contreparties. Le bord de la roue est arrondi, ce qui change très
peu la forme des collisions latérales. La masse d'une roue passe de 0,157 à 0,105 kg,
car elle est calculée à partir du volume.

### Cône de frottement elliptique

Avec le cône pyramidal, qui est le défaut de MuJoCo, l'enfoncement des roues dans le
sol dépend de μ :

| μ | Enfoncement, cône pyramidal | Enfoncement, cône elliptique |
|---|---|---|
| 0,2 | 0,008 mm | 0,336 mm |
| 0,5 | 0,060 mm | 0,336 mm |
| 1,0 | 0,336 mm | 0,336 mm |

Changer le frottement changeait donc aussi le rayon effectif des roues. C'est un
artefact du modèle de contact, gênant pour l'expérience E3 (frottement divisé par
deux) et pour la randomisation de domaine. Le cône elliptique rend la géométrie du
contact indépendante de μ, et il est même plus rapide sur ce modèle (environ
156 000 pas/s contre 111 000 dans l'environnement de test, à remesurer sur le poste de
travail).

### Contact avec le sol

Le sol est ramené à z = 0. Il était à z = −0,5 m, ce qui faisait tomber le robot de
55 cm à chaque réinitialisation. Les paramètres `solref` et `solimp` du test de chute
libre sont supprimés : ils avaient été choisis pour une boîte qui tombe, pas pour des
roues qui roulent.

Avec le contact par défaut, il reste l'écart de 0,34 % en ligne droite. Il vient de
l'enfoncement du contact mou, δ = 0,34 mm, qui réduit le rayon effectif : la mesure
correspond à r<sub>eff</sub> ≈ r − δ/2. Un sol plus raide (`solref="0.005 1"`)
ramène l'écart à 0,15 %, mais ce n'est pas nécessaire, car on est loin du seuil de
2 %. Cet écart est une erreur systématique de l'odométrie, à garder en tête pour E1.

### Intégrateur et pas de temps

| Δt | Euler | implicitfast |
|---|---|---|
| 0,5 ; 1 ; 2 ms | stable | stable |
| 4 ; 8 ms | instable, même à l'arrêt | stable, mêmes mesures |

L'actionneur `velocity` est un amortisseur visqueux de coefficient k<sub>v</sub>.
Euler le traite de façon explicite, ce qui impose Δt < 2 I<sub>eff</sub>/k<sub>v</sub>.
Au sol, chaque roue entraîne la moitié de la masse du robot, donc
I<sub>eff</sub> ≈ I<sub>roue</sub> + (M/2)·r² ≈ 1,7·10⁻³ kg·m². On obtient
Δt < 3,4 ms, ce qui est cohérent avec la frontière observée entre 2 et 4 ms.

Une roue en l'air n'a plus que sa propre inertie, et la limite tombe alors à 0,26 ms.
Or les roues décollent brièvement lors des démarrages brutaux (voir plus bas). Avec
Euler, il faudrait donc un pas de temps près de dix fois plus petit pour être sûr.
`implicitfast` intègre cet amortissement implicitement, et c'est la raison de ce
choix.

Le pas de temps de 2 ms correspond à 50 pas physiques par décision à 10 Hz. Les
mesures cinématiques n'y sont pas sensibles. Le comportement lors des collisions avec
les obstacles reste à vérifier au jour 5.

### Gain k<sub>v</sub>

| k<sub>v</sub> | 0,1 | 0,3 | 1 | 3 | 10 |
|---|---|---|---|---|---|
| Écart actionneur, cercle | 1,26 % | 0,42 % | 0,13 % | 0,04 % | 0,01 % |

En ligne droite et en rotation sur place, l'erreur de l'actionneur est nulle. Sur le
cercle, elle suit exactement 1/k<sub>v</sub> : c'est l'erreur statique d'un
correcteur proportionnel soumis à un couple permanent, τ = k<sub>v</sub>·Δω. Ce couple
vient vraisemblablement du centre de masse, situé environ 5 cm en arrière de
l'essieu : en virage, le moment de la force centripète doit être compensé par une
différence de poussée entre les roues.

Avec k<sub>v</sub> = 1, l'erreur est de 0,13 %, ce qui est négligeable. Lors des
transitoires, c'est la saturation du couple (0,2 N·m) qui domine dès que l'écart de
vitesse dépasse 0,2 rad/s, et non le gain. On garde donc k<sub>v</sub> = 1.

### Frottement μ et glissement

Une roue patine dès que le couple moteur maximal dépasse ce que le frottement peut
transmettre. Au repos, cela donne un seuil
μ\* = τ<sub>max</sub>/(r·N) = 0,2/(0,05 × 4,49) ≈ 0,89. Pendant l'accélération, une
partie de la charge passe sur la roulette, ce qui élève encore ce seuil.

![Glissement au démarrage selon μ](../figures/glissement_mu.png)

Sur la figure, le robot passe de l'arrêt à pleine commande (10 rad/s sur les deux
roues). Les accélérations moyennes mesurées se comparent aux prédictions ainsi :

| μ | Accélération mesurée | Prédiction | Limite |
|---|---|---|---|
| 0,2 | 1,34 m/s² | 1,43 m/s² | adhérence, μ·2N/M |
| 0,5 | 2,76 m/s² | 3,56 m/s² | adhérence, μ·2N/M |
| 1,0 | 5,30 m/s² | 5,86 m/s² | moteur, 2τ/(r·(M + 2I/r²)) |

À μ = 0,2, les roues atteignent leur vitesse maximale presque immédiatement et le
robot accélère à la limite d'adhérence ; le glissement ne disparaît qu'après 0,4 s.
À μ = 0,5, l'écart à la prédiction s'explique par le sautillement décrit plus bas :
roues en l'air 21 % du temps, et 3,56 × 0,79 ≈ 2,81 m/s². À μ = 1, le robot suit les
roues après un court patinage initial et atteint 0,45 m/s en 0,09 s.

Commentaire en deux phrases : *sous le seuil μ\* ≈ 0,9, le couple moteur dépasse ce
que le frottement peut transmettre, si bien que les roues patinent et que
l'accélération du robot est limitée par l'adhérence, et non par le moteur. En régime
établi, le glissement disparaît quel que soit μ, ce qui explique que les validations
à vitesse constante soient insensibles au frottement.*

On garde μ = 1,0, l'ordre de grandeur du caoutchouc sur un sol dur. Avec E3
(μ = 0,5), les démarrages brutaux patineront franchement : c'est précisément ce que
cette perturbation doit tester.

## Points ouverts

**Sautillement au démarrage brutal.** Lors d'un démarrage à pleine commande, les roues
quittent le sol par intermittence, de 0,2 mm environ. Cela représente 21 % du temps
pendant les 0,3 premières secondes à μ = 0,5, et 3 % à μ = 1. C'est ce qui donne les
marches d'escalier sur la figure. Ni `impratio = 10` ni un contact plus raide ne
réduisent ce phénomène. Il disparaît en régime établi. Pistes : limiter
l'accélération dans `control_node` (jour 9), ce que fait déjà la fenêtre dynamique du
DWA, puis vérifier si PPO, qui peut commander brutalement, en souffre.

**Contrôles du jour 2 à refaire avec le nouveau modèle.** La boîte du châssis est
décalée de 5 cm vers l'arrière. L'inertie du corps `chassis` est donc calculée autour
du centre de masse commun à la boîte et à la roulette : il faut la comparer au calcul
à la main avec le théorème de Huygens. Le critère « moins de 1 mm en 10 s à commande
nulle » est aussi à revérifier.

**Collisions avec les obstacles.** Elles n'ont pas été testées ici. Elles restent à
valider au jour 5, avec le même pas de temps.
