# Kaggle retrievial 

# in a google colab : 

!pip install kaggle
import os
os.environ['KAGGLE_CONFIG_DIR'] = '/content/.kaggle'
!mkdir -p /content/.kaggle

# Upload your kaggle.json file
from google.colab import files
files.upload()  # choose kaggle.json

!mv kaggle.json /content/.kaggle/
!chmod 600 /content/.kaggle/kaggle.json

# Télécharger le dataset
!kaggle datasets download -d author/database_name  # create zip untitled : database_name.zip

!unzip database_name.zip -d data/database_name

# Verify content
import os 
os.listdir('data/database_name')[:10]


# if saving is needed
from google.colab import drive
drive.mount('/content/drive')
!cp -r data/boxes /content/drive/MyDrive/colis_dataset/