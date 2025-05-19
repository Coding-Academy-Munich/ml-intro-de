# %%
# %% [markdown]
#
# <div style="text-align:center; font-size:200%;">
#  <b>Workshop Broadcasting</b>
# </div>
# <br/>
# <div style="text-align:center;">Dr. Matthias Hölzl</div>
# <br/>

# %% [markdown]
#
# # Workshop: NumPy Universelle Operationen und Broadcasting
#
# In diesem Workshop werden wir uns universelle Operationen und Broadcasting
# genauer ansehen. Einige der Operationen, die in diesem Workshop verwendet
# werden, wurden in den Vorlesungen nicht vorgestellt, Sie müssen in die [NumPy
# Dokumentation](https://numpy.org/doc/stable/reference/ufuncs.html) schauen, um
# sie zu entdecken.

# %%
import numpy as np

# %%
arr1 = np.arange(1, 25).reshape(2, 3, 4)
lst1 = [2, 3, 5, 7]

# %% [markdown]
#
# ## Universelle Operationen
#
# Berechnen Sie Arrays `arr2` und `arr3`, die die Elemente von `arr1` bzw.
# `lst1` quadriert enthalten.

# %% [markdown]
#
# Berechnen Sie das Produkt von `arr1` und `lst1`. Bevor Sie Ihre Lösung
# auswerten: Versuchen Sie, die Form des Ergebnisses zu bestimmen. Wie wird die
# Form des Ergebnisses bestimmt? Benötigen Sie eine universelle Funktion oder
# können Sie die Multiplikation einfach als normales Produkt durchführen?

# %% [markdown]
#
# Schreiben Sie eine Funktion `may_consume_alcohol(ages)`, die eine Liste oder
# ein 1-dimensionales Array von Altersangaben entgegennimmt und ein Array
# zurückgibt, das die Werte `"no"` enthält, wenn der entsprechende Index im
# Eingabearray kleiner als 18 ist, `"maybe"`, wenn der Wert über 18 aber
# kleiner als 21 ist und `"yes"`, wenn der Wert mindestens 21 ist.
#
# Zum Beispiel gibt `may_consume_alcohol([15, 20, 30, 21, 20, 17, 18])` ein
# Array zurück, das `['no', 'maybe', 'yes', 'yes', 'maybe', 'no', 'maybe']`
# enthält.

# %% [markdown]
#
# Schreiben Sie eine Funktion `double_or_half(values)`, die eine Liste oder ein
# 1-dimensionales Array von Zahlen entgegennimmt und ein Array zurückgibt, das
# die Werte `v * 2` enthält, wenn das entsprechende Element von `values` ungerade
# ist und `v // 2`, wenn das entsprechende Element gerade ist.
#
# Zum Beispiel sollte `double_or_half([0, 1, 2, 5, 10, 99])` ein Array
# zurückgeben, das die Werte `[0, 2, 1, 10, 5, 198]` enthält.
#
# *Hinweis:* Schauen Sie in die Dokumentation für die Funktion `choose`.
