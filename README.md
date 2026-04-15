# LLM-course
This repo contain asignment and implementation from the course Large Lenguage Model.
The repo contain just the implemantarion and description of every assignment without theoretacal information.

# Structure
## Assignment-1
* attention.py
* lm.py
* main.py
* mlp.py
* tests.py
* tramsformer.py

## Assignment-2


# How To Run In Google Colab
* 1. Open Colab

Go to: https://colab.research.google.com
Create a new notebook.

2. Clone the repository
`!git clone https://github.com/gavZich/LLM-course.git`

3. Navigate to the project
`%cd /content/LLM-course`
`!ls`

`%cd /content/LLM-course/Assignment-1/code`
`!ls`

4. Set GPU: Runtime → Change runtime type → GPU.

5. Run tests
`python tests.py`

6. Run training
`!python main.py`

🔁 Updating code after changes

If you update code in VSCode:

`%cd /content/LLM-course`
`!git pull`