03/11/2026 :

Pour le moment je me suis concentré plutôt sur le projet de simulation de fluide. Pour le projet robotique, j'ai simplement créé le dépôt Github et l'arborescence et j'ai installé (ré-installé) les extensions nécessaires : python -c 'import rclpy, mujoco, torch, gymnasium, stable_baselines3' fonctionne.

Création de sandbox_free_fall.py : enregistrement de la hauteur, comparaison entre les valeurs prévus par Mujoco, un schéma RK4 et Euler.
04/11/2026 :

Dernière installation du setup et fixe de certains bug qu'il y avait
Création du robot (avec un petit script de déplacement)
Création de l'arena.xml qui regroupe le robot et le sol (pour l'instant très simple)

09/11/2026 : 
Bon comme j'ai pris bcp de retard, j'ai codé bcp avec Claude pour que le programme soit bon au temps voulu. Désormais on dispose d'un environnement viable, dans lequel le robot peux évoluer (et j'ai vérifié : ça compile). 

Problème : je comprends rien au code (normal j'ai rien fait lol), donc je me donne aujourd'hui pour le relire et bien le comprendre. Je ferais au fur et à mesure de ma relecture de la lecture d'article sur lesquels tout ça est basé, et je vais effectuer des tests dans un notebooks (qu'on peut retrouver dans le dossier notebooks, il s'appelle 'draft.ipynb') pour vérifier que je comprends bien tout ce qu'il se passe.
