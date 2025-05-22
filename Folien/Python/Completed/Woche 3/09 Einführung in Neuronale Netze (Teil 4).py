# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Einführung in Neuronale Netze (Teil 4)</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %%
import torch
import torch.nn as nn


# %% [markdown]
# ## Modelle
#
# <img src="img/Figure-11-001.png" style="width: 100%;"/>

# %% [markdown]
# ## Für Neuronale Netze:
#
# Was repräsentiert werden kann hängt ab von
#
# - Anzahl der Layers
# - Anzahl der Neutronen per Layer
# - Komplexität der Verbindungen zwischen Neutronen

# %% [markdown]
# ### Was kann man (theoretisch) lernen?
#
# Schwierig aber irrelevant

# %% [markdown]
# ### Was kann man praktisch lernen?
#
# Sehr viel, wenn man genug Zeit und Daten hat

# %% [markdown]
# ### Was kann man effizient lernen?
#
# Sehr viel, wenn man sich geschickt anstellt
# (und ein Problem hat, an dem viele andere Leute arbeiten)

# %% [markdown]
# # Bias/Variance Tradeoff
#
# - Modelle mit geringer Expressivität (representational power)
#   - Können schnell trainiert werden
#   - Arbeiten mit wenig Trainingsdaten
#   - Sind robust gegenüber Fehlern in den Trainingsdaten
#
# - Wir sind nicht an einer möglichst exakten Wiedergabe unserer Daten interessiert
#
# - Entscheidend ist wie gut unser Modell auf unbekannte Daten generalisiert

# %% [markdown]
# <img src="img/Figure-09-002.png" style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# <img src="img/Figure-09-004.png" style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# <img src="img/Figure-09-003.png" style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# <img src="img/Figure-09-005.png" style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
#
# ### Generalisierung und Rauschen
# <img src="img/Figure-09-008.png" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# <img src="img/Figure-09-009.png" style="width: 80%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# <img src="img/Figure-09-010.png" style="width: 40%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Komplexität der Entscheidungsgrenze
#
# <img src="img/Figure-09-006.png" style="width: 100%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# <img src="img/Figure-09-001.png" style="width: 50%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Datenverteilung und Qualität
#

# %% [markdown]
# ### Erinnerung: die Trainings-Schleife
#
# <img src="img/Figure-08-001.png" style="width: 20%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# <img src="img/Figure-08-001.png" style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Was lernt ein Klassifizierer?
#
# <img src="img/Figure-08-002.png" style="width: 60%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# <img src="img/Figure-08-003.png" style="width: 100%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# <img src="img/Figure-08-004.png" style="width: 70%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# # Wie gut sind wir?
#
# Wie wissen wir, wie gut unser Modell wirklich ist?

# %% [markdown]
# ## Was kann schief gehen?
#
# <img src="img/Figure-03-015.png" style="width: 100%; margin-left: auto; margin-right: auto; 0"/>

# %% [markdown]
# ## Was kann schief gehen?
#
# <img src="img/Figure-03-017.png" style="width: 100%; margin-left: auto; margin-right: auto; 0"/>

# %% [markdown]
# ## Was kann schief gehen?
#
# <img src="img/Figure-03-018.png" style="width: 80%; margin-left: auto; margin-right: auto; 0"/>

# %% [markdown]
# ## Accuracy: Wie viel haben wir richtig gemacht?
#
#
# <img src="img/Figure-03-023.png" style="width: 60%; margin-left: auto; margin-right: auto; 0"/>

# %% [markdown]
# ## Precision: Wie gut sind unsere positiven Elemente?
#
#
# <img src="img/Figure-03-024.png" style="width: 60%; margin-left: auto; margin-right: auto; 0"/>

# %% [markdown]
# ## Recall: Wie viele positive Elemente haben wir übersehen?
#
#
# <img src="img/Figure-03-026.png" style="width: 60%; margin-left: auto; margin-right: auto; 0"/>
