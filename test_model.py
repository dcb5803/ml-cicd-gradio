from app.model import train_model

def test_model_output():
    clf, iris = train_model()
    assert clf.predict([iris.data[0]]).shape == (1,)
