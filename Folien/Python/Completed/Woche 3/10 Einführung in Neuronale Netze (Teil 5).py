# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Einführung in Neuronale Netze (Teil 5)</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
# ## Data Engine (Tesla)
#
# <img src="img/data-engine.jpeg" style="width: 100%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# ## Probleme: Abhängigkeiten
#
# Relevante Informationen sind nicht immer nahe in den Daten:
#
# "Er hatte mit dem Mann, der ihm den Schlüssel, der zum Schloss, das ihn von großem Reichtum trennte, gehörte, gab, noch nicht gesprochen.

# %% [markdown]
# # Memory / State
# <img src="img/Figure-22-012.png" style="width: 20%; margin-left: auto; margin-right: auto;"/>

# %% [markdown]
# Funktioniert gut aber mit gewissen Schwächen.
#
# Man muss wissen, welche Information für das aktuell betrachtete Element relevant ist:

# %% [markdown]
# - The cat didn't cross the street because *it* was too wide.

# %% [markdown]
# <img src="img/garfield.jpg" style="float: right;width: 60%;"/>

# %% [markdown]
# <img src="img/garfield-yawn.png" style="float: right;width: 60%;"/>
#
# - The cat didn't cross the street because *it* was too tired.

# %% [markdown]
# - The cat didn't cross the street because *it* was too wet.
#
# <img src="img/garfield-rain4.jpg" style="float: right;width: 60%;"/>

# %% [markdown]
# - The cat didn't cross the street because *it* was raining.
#
# <img src="img/garfield-rain2.gif" style="float: right;width: 60%;"/>

# %% [markdown]
# # The Bitter Lesson (Rich Sutton)
#
# [T]he only thing that matters in the long run is the leveraging of computation.
#
# Corollary: And data. Lots of data.

# %%
