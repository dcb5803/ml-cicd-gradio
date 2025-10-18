import gradio as gr
from app.model import train_model

clf, iris = train_model()

def predict(sepal_length, sepal_width, petal_length, petal_width):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    return iris.target_names[clf.predict(input_data)[0]]

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Sepal Length"),
        gr.Number(label="Sepal Width"),
        gr.Number(label="Petal Length"),
        gr.Number(label="Petal Width")
    ],
    outputs="text",
    title="Iris Classifier"
)
