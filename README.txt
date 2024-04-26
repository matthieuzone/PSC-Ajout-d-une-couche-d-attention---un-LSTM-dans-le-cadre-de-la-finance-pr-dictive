DEPENDANCES : de nombreuses librairies python sont nécessaire pour faire fonctionner notre code. Il est nécessaire d'installer celles manquantes lorsque l'on rencontre des erreurs d'importations

Pour entrainer et tester différents modèles, utiliser le fichier all_models.ipynb

Pour les algorithmes de feature selection, utiliser le fichier feature_selection.ipynb

Pour l'optimisation d'hyperparamètres (Population based training, random search), dans le dossier PBT, les différents fichier .py ne sont utilisés par PBT_pytorch.ipynb, qu'il faut utiliser pour lancer des experimentations (attention, temps de calcul très long : plusieures heures)
le fichier backtest.ipynb permet ensuite de tester un modèle à l'aide des outils de backtest

IMPORTANT : pour que PBT_pytorch.ipynb fonctionne, il faut créer un dossier 'ray_checkpoints' et 'ray_results' à la racine du disque dur, s'il ne le fait pas tout seul.