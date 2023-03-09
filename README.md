# ML_tutorials
Machine learning tutorials for NSCI0028

In this tutorial, we introduce first the basics to start using Python for data mining and machine learning.

The student should follow the following steps:

1- Start by copying the link to the Jupyter Notebook from the folder in GitHub entitled "Basics_in_Python.ipynb"

2- Open Colab in the link below:
https://colab.research.google.com/

3- Select the tab Github and paste the link in the dedicated field to search for the Notebook

4- Open the link of the Notebook and press Connect

5- Each time, you would like to run an exercise, press on the small triangle in the black circle and wait for a few second to see the output underneath the code.

6- When you finish all the exercice, open a fresh colab session and choose another notebook.

7- Follow this order of the notebooks:

      a- Basics_in_Python.ipynb

      b-Introduction_to_NumPy.ipynb

      c-Introduction_to_Pandas.ipynb

      d-Basics_visualisation.ipynb

      e-Loops and functions.ipynb

      f-Preprocessing.ipynb
      
      g-Preprocessing_Iris_dataset.ipynb
     
      h-Feature_selection.ipynb.ipynb
      
      i-Underfitting_Overfitting.ipynb
      
      j-Outliers_removal.ipynb

      k-Multiple_Linear_Regression.ipynb
     
      l-Random_Forest_Iris.ipynb
      
      m-KNN.ipynb
      
      n-KNN-optimsation.ipynb
      
      o-SVM.ipynb
      
      p-imbalanced_classes.ipynb
        
      q-Kmeans.ipynb
      
      r-kmeans_2.ipynb
      
      s-hierarchical_clustering.ipynb
      
      t-PCA_1.ipynb
      
      u-PCA_2.ipynb
      
      v-PCA_3.ipynb
      
      w-Logistic_regression.ipynb
      
Install and run Knime
      
0) Create a directory to install it in eg:

NOTE: your ssh connection to Aristotle will need X tunnelling enabled. So, run Xming or Exceed then run Putty and activate X11.
1) Run the following:
mkdir -p ~/apps/KNIME
cd ~/apps/KNIME

2) Download the 64bit Linux version, unarchive it and update your PATH:
wget https://download.knime.org/analytics-platform/linux/knime-latest-linux.gtk.x86_64.tar.gz
tar -xvf knime-latest-linux.gtk.x86_64.tar.gz
export PATH=$PATH:~/apps/KNIME/knime_4.7.1

3) Load the following modules:

module -f unload compilers mpi gcc-libs
module load beta-modules
module load gcc-libs/10.2.0

4) un it using:

knime &




