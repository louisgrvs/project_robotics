03/11/2026 :

Pour le moment je me suis concentré plutôt sur le projet de simulation de fluide. Pour le projet robotique, j'ai simplement créé le dépôt Github et l'arborescence et j'ai installé (ré-installé) les extensions nécessaires : python -c 'import rclpy, mujoco, torch, gymnasium, stable_baselines3' fonctionne.

Création de sandbox_free_fall.py : enregistrement de la hauteur, comparaison entre les valeurs prévus par Mujoco, un schéma RK4 et Euler.
04/11/2026 :

Dernière installation du setup et fixe de certains bug qu'il y avait
Création du robot (avec un petit script de déplacement)
Création de l'arena.xml qui regroupe le robot et le sol (pour l'instant très simple)
