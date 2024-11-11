# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Huggingface</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# - [https://huggingface.co/](https://huggingface.co/)
# - Startup, das viele Modelle für NLP, CV und mehr bereitstellt
# - Python-Bibliothek `transformers` für vortrainierte Modelle
# - Datasets-Bibliothek für Datensätze
# - Viele weitere Bibliotheken

# %% [markdown]
#
# ## Huggingface Transformers
#
# - Bibliothek für vortrainierte Modelle und Tokenizer
# - NLP, CV, Audio, Multimodal
# - PyTorch, TensorFlow, JAX
# - Pipelines für einfache Anwendung
# - Vollständiges API für weitergehende Anpassungen

# %% [markdown]
#
# ## Huggingface Datasets
#
# - Bibliothek für Datensätze
# - Einfaches Laden und Verwenden von Datensätzen
# - Viele Datensätze für NLP, CV, Audio, Multimodal

# %% [markdown]
#
# ## Beispiele für Pipelines
#
# ### Spracherkennung
#
# - [Martin Luther King Jr. Speech](https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac)
# - Wir wollen eine Transkription des Audios

# %%
import torch
from transformers import pipeline

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
asr = pipeline(task="automatic-speech-recognition", device=device)

# %%
audio_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"

# %%
transcription = asr(audio_url)

# %%
transcription

# %%
asr = pipeline(task="automatic-speech-recognition",
               model="openai/whisper-large-v3",
               device=device)

# %%
transcription = asr(audio_url)

# %%
transcription

# %% [markdown]
#
# ### Sentimentanalyse
#
# - Wir wollen die Stimmung eines Textes bestimmen

# %%
nlp = pipeline(task="sentiment-analysis",
               model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
               device=device)

# %%
text = "I love Huggingface!"

# %%
sentiment = nlp(text)

# %%
sentiment

# %%
texts = [
    "I love Huggingface!",
    "I hate Huggingface!",
    "I am neutral about Huggingface.",
    "The people at Huggingface are great!",
    "Machine learning does have its limitations."
]

# %%
nlp(texts)

# %%
nlp = pipeline(task="sentiment-analysis",
               model="facebook/bart-large-mnli",
               device=device)

# %%
nlp(texts)

# %% [markdown]
#
# ### Zero-Shot-Klassifikation
#
# - Wir wollen Texte klassifizieren, ohne Trainingsdaten zu haben
# - Wir geben Klassen vor, die das Modell verwenden soll

# %%
classifier = pipeline(task="zero-shot-classification",
                      model="facebook/bart-large-mnli",
                      device=device)

# %%
text = "I am looking for a new car."

# %%
classes = ["business", "politics", "entertainment", "sports"]

# %%
classifier(text, classes)

# %%
texts = [
    "I am looking for a new car.",
    "The new movie is great!",
    "The company is doing well.",
    "The team won the championship.",
    "Dow Jones heading for new record.",
    "Marvel's latest release flopped."
]

# %%
classifier(texts, classes)

# %%
classifier(texts, ["positive", "negative"])

# %% [markdown]
#
# ### Textgenerierung
#
# - Wir wollen Texte generieren
# - Wir geben einen Anfangstext vor

# %%
generator = pipeline(task="text-generation",
                     model="HuggingFaceTB/SmolLM2-360M",
                     device=device)

# %%
prompts = ["Once upon a time, "] * 10

# %%
generator(prompts)

# %%
generator = pipeline(task="text-generation",
                     model="HuggingFaceTB/SmolLM2-360M-Instruct",
                     device=device)

# %%
generator(prompts)


# %% [markdown]
#
# ## Zero-Shot Bildklassifikation
#
# - Wir wollen Bilder klassifizieren
# - Wir geben ein Bild und die Klassen vor

# %%
classifier = pipeline(task="zero-shot-image-classification",
                      model="google/siglip-so400m-patch14-384")

# %%
image_url = "https://huggingface.co/datasets/Narsil/image_dummy/resolve/main/elephant.jpg"

# %%
classifier(image_url, candidate_labels=["elephant", "dog", "cat", "bird"])
