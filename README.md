# Python-scripts
 Collection of python  projects, Machine Learnings, Time Series analysis and  Data Visualization.
Machine Learning Notebooks
This project aims at teaching you the fundamentals of Machine Learning in python. It contains the example code and solutions to the exercises in my O'Reilly book Hands-on Machine Learning with Scikit-Learn and TensorFlow:



Warning: there is now a newer edition of this book, please check out github.com/ageron/handson-ml2.

Quick Start
Want to play with these notebooks online without having to install anything?
Use any of the following services.

WARNING: Please be aware that these services provide temporary environments: anything you do will be deleted after a while, so make sure you download any data you care about.

Recommended: open this repository in Colaboratory: 

Or open it in Binder: 

Note: Most of the time, Binder starts up quickly and works great, but when handson-ml is updated, Binder creates a new environment from scratch, and this can take quite some time.
Or open it in Deepnote: 

Just want to quickly look at some notebooks, without executing any code?
Browse this repository using jupyter.org's notebook viewer: 

Note: github.com's notebook viewer also works but it is slower and the math equations are not always displayed correctly.

Want to run this project using a Docker image?
Read the Docker instructions.

Want to install this project on your own machine?
Start by installing Anaconda (or Miniconda), git, and if you have a TensorFlow-compatible GPU, install the GPU driver, as well as the appropriate version of CUDA and cuDNN (see TensorFlow's documentation for more details).

Next, clone this project by opening a terminal and typing the following commands (do not type the first $ signs on each line, they just indicate that these are terminal commands):

$ git clone https://github.com/ageron/handson-ml.git
$ cd handson-ml
Next, run the following commands:

$ conda env create -f environment.yml
$ conda activate tf1
$ python -m ipykernel install --user --name=python3
Finally, start Jupyter:

$ jupyter notebook
If you need further instructions, read the detailed installation instructions.

FAQ
Which Python version should I use?

I recommend Python 3.7. If you follow the installation instructions above, that's the version you will get. Most code will work with other versions of Python 3, but some libraries do not support Python 3.8 or 3.9 yet, which is why I recommend Python 3.7.

I'm getting an error when I call load_housing_data()

Make sure you call fetch_housing_data() before you call load_housing_data(). If you're getting an HTTP error, make sure you're running the exact same code as in the notebook (copy/paste it if needed). If the problem persists, please check your network configuration.

I'm getting an SSL error on MacOSX

You probably need to install the SSL certificates (see this StackOverflow question). If you downloaded Python from the official website, then run /Applications/Python\ 3.7/Install\ Certificates.command in a terminal (change 3.7 to whatever version you installed). If you installed Python using MacPorts, run sudo port install curl-ca-bundle in a terminal.

I've installed this project locally. How do I update it to the latest version?

See INSTALL.md

How do I update my Python libraries to the latest versions, when using Anaconda?

See INSTALL.md

Contributors
I would like to thank everyone who contributed to this project, either by providing useful feedback, filing issues or submitting Pull Requests. Special thanks go to Haesun Park and Ian Beauregard who reviewed every notebook and submitted many PRs, including help on some of the exercise solutions. Thanks as well to Steven Bunkley and Ziembla who created the docker directory, and to github user SuperYorio who helped on some exercise solutions.
