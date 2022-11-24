# 0 - Lancez un run avec salloc (1 noeud et quelques cpu suffisent) : salloc -p routage --time=6:00:00
# 1 - connectez vous sur votre noeud (squeue)
# 2 - Installez une distribution python : Exécutez le script install_miniconda.sh situé à la racine des TPs
# 3 - activez anaconda : source ~/anaconda/bin/activate
# 4 - Créez un environnement dédié au tp : conda create -y --name tp1 python=3.10 pip
# 5 - activez l'environnement : conda activate tp1

# --- JupyterLab ---
# 6 - installez les modules nécessaires au TP : pip install -r requirements.txt 
# 7 - lancez jupyter-lab : jupyter-lab --ip=0.0.0.0
# 8 - identifiez le port sur lequel il s'est lancé (normalement 8888) ou précisez --port 8888
# 9 - depuis votre machine locale faites un port forwarding pour accéder au port (8888) de votre noeud : ssh -L 8888:mirielXXX:8888 formation
# 10 - connectez vous depuis votre navigateur web sur : 127.0.0.1:8888 (ou votre port du #7)
# 11 - C'est parti !


# --- VSCode ---
# 6 Configurez .ssh/config de la sorte
host formation
   user jacq
   Hostname plafrim
   ProxyCommand ssh -W %h:%p -q %u@formation.plafrim.fr
   ForwardAgent yes
   IdentityFile ~/.ssh/pi_rsa

host miriel*
   user jacq
   ProxyCommand ssh -W %h:%p -q formation
   ForwardAgent yes
   IdentityFile ~/.ssh/pi_rsa
# 7 installez l'extension pack sur votre host : ms-vscode-remote.vscode-remote-extensionpack
# 8 Connectez votre VSCode en ssh sur le miriel alloué en 0
# 9 Installez l'extension python : ms-python.python
# 10 F1 : "select interpreter to start jupyter" Sélectionnez conda du tp1
# 11 installez les modules nécessaires au TP : pip install -r requirements-vscode.txt 
# 12 - C'est parti ! 
