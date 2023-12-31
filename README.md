<h1> MAIR_projectG16 </h1>
 
Welcome to the Restaurant Recommendations Dialog System project of group 16! 
In this project, we will design, implement, evaluate, and document a restaurant recommendations dialog system using various AI techniques. Our goal is to create a AI-driven system that can assist users in finding a restaurant for their needs.

<h2> Table of Contents </h2>

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Libraries](#libraries)
- [License](#license)

<h2> Introduction</h2>
This project is divided into two main parts:

**Part 1**: Implementation of the Dialog System

In the first part of the project, we will focus on building the foundation of our dialog system. This involves:

- **Domain Modeling**: 
We will model a specific knowledge domain related to restaurant recommendations, ensuring our system understands the context.

- **Machine Learning for NLP**: 
We'll implement and empirically evaluate a machine learning classifier for natural language processing (NLP). This will enable our system to understand and interpret user queries effectively and with relatively high precision.

- **Dialog System Application**: 
Using the dialog model and classifiers, we'll develop a text-based dialog system application. Users will interact with our system to receive restaurant recommendations.

The conversation flow of the dialog system application, where the rectangles represent system utterances and circles represent 	user responses. The rhombus shape represents an internal logic handler of inputs ( if-else functionality).
![Diagram](diagram.png)

**Part 2**:
Has not started yet.

<h2> Installation</h2>

Make sure you have Python installed on your system before running this command.
`pip install -r requirements.txt`

This command automatically installs the required libraries on your machine.

<h2>Usage</h2>
Run main.py to execute the finite-state Machine of part 1.b, run text_classification.py for part 1.a.

*We need to explain how users can use our project effectively. Provide examples and instructions as needed.*

<h2>Libraries</h2>

The project relies on several Python libraries to achieve its functionality. 
Here are all the libraries used in this project:

- **pandas**: 
Pandas is a powerful data manipulation and analysis library that simplifies working with structured data.
- **pickle**: 
Pickle is a simple library that turns any object into raw data so it can be used later.
- **numpy**: 
NumPy is a fundamental library for numerical computing, enabling efficient operations on arrays and matrices.
- **scikit-learn**: 
Scikit-learn is a comprehensive machine learning library, providing tools for data preprocessing, model selection, and evaluation.
- **TensorFlow and Keras**: 
TensorFlow is a machine learning framework, while Keras is a high-level API running on top of TensorFlow. They are used for deep learning tasks, including building and training neural networks.
- **matplotlib**: 
Matplotlib is a versatile data visualization library, allowing you to create various types of plots and charts.
- **LabelEncoder**: 
LabelEncoder, from scikit-learn, is used for encoding categorical labels as integers.
- **TfidfVectorizer**: 
TfidfVectorizer, also from scikit-learn, is used for text feature extraction, particularly in natural language processing (NLP) tasks.

Please refer to the documentation of each library for more details on their usage and capabilities.

<h2>License</h2>

*We need to add a License to our project*