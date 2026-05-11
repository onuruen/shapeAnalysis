# Morphological Object Detection with Hit-or-Miss Transform

(Deutsche Fassung unten)

A computer vision exercise exploring morphological operations and the **Hit or Miss Transform (HMT)** for shape specific object detection and sequential pixel manipulation using OpenCV.

---

## Overview

This script processes a binary image (`Shapes1.bmp`) containing three distinct shape types **squares**, **L-shapes**, and **hollow circles** and isolates each class using a pipeline of morphological operations. The core idea is to progressively subtract detected shapes from the original image until only the target objects remain.

---

## Pipeline

```
Original Image
      │
      ├──[HMT: square kernel]──► Square locations
      │         │
      │    [Dilate & subtract]
      │         │
      ▼         ▼
 Hollow + L-shapes
      │
      ├──[HMT: L-shape kernel]──► L-shape locations
      │         │
      │    [Reconstruct mask & subtract]
      │         │
      ▼         ▼
  Hollow Circles Only
```

---

## Key Concepts

### Binary Representation

The grayscale image is thresholded and converted to a `{0, 1}` binary image (`binary_01`) for use with OpenCV's morphological functions, which expect this format for HMT operations.

### Hit-or-Miss Transform (HMT)

HMT is a morphological operation that detects **exact local patterns** in a binary image. The kernel encodes three states:

| Value | Meaning                                 |
| ----- | --------------------------------------- |
| `+1`  | Foreground pixel must be present (hit)  |
| `-1`  | Background pixel must be present (miss) |
| `0`   | Don't care                              |

A pixel location is marked as a match only when **all** hit conditions and **all** miss conditions are satisfied simultaneously. This makes HMT far more selective than standard erosion or dilation.

```python
cv2.morphologyEx(binary_01, cv2.MORPH_HITMISS, kernel, anchor=(r, c))
```

---

## Sections

### Section 1 — Square Detection & Elimination

A 6×6 kernel with a solid `+1` interior and `-1` border is used to detect the **top-left anchor point** of each square. The anchor is set to `(1, 1)` to align the interior region correctly.

```
[-1, -1, -1, -1, -1, -1]
[-1,  1,  1,  1,  1, -1]
[-1,  1,  1,  1,  1, -1]
[-1,  1,  1,  1,  1, -1]
[-1,  1,  1,  1,  1, -1]
[-1, -1, -1, -1, -1, -1]
```

Once locations are found, each square's pixel region `[r:r+4, c:c+4]` is painted into a mask, which is subtracted from the original — leaving only L-shapes and hollow circles.

![Square Analysis](result-images/square-analysis.PNG)

---

### Section 2 — L-Shape Detection & Elimination

A 6×7 kernel detects the **L-shape pattern** (filled rectangle with a missing bottom-right corner). The anchor is at `(0, 0)` so the detection point corresponds to the top-left of the shape.

```
[ 1,  1,  1,  1,  1,  1,  1]
[ 1,  1,  1,  1,  1,  1,  1]
[ 1,  1,  1,  1,  1,  1,  1]
[ 1,  1,  1,  1,  1,  1,  1]
[ 1,  1,  1,  1, -1, -1, -1]
[ 1,  1,  1,  1, -1, -1, -1]
```

Rather than using HMT output directly as a mask (which would only mark single pixels), the L-shape geometry is **reconstructed** by painting the correct pixel region for each detected location using a binary shape mask. This "dilation by shape" approach recovers the full object footprint before subtraction.

![L-shape Analysis](result-images/L-shapeAnalysis.PNG)

---

### Section 3 — Hollow Circle Isolation

After the two subtraction steps, hollow circles are obtained by:

```python
hollow_only = cv2.subtract(diff_no_squares, recovered_l_shapes_255)
```

![Hollow Circles](result-images/hollowCircles.PNG)

This works because `diff_no_squares` already contains only L-shapes and hollow circles, so subtracting the reconstructed L-shape mask leaves the hollow circles in isolation.

---

## Observations

- **HMT is highly sensitive to kernel design.** A kernel that is even slightly misaligned with the actual shape dimensions in the image will produce zero detections. Correct anchor placement is equally critical.
- **HMT output marks anchor points, not full shapes.** To eliminate detected objects, their pixel regions must be explicitly reconstructed and painted into a subtraction mask.
- **Sequential subtraction** is an effective strategy when shapes are non-overlapping and their kernels are well-defined.
- `cv2.subtract` handles clipping automatically (no negative values), making it safe for mask-based elimination on `uint8` images.

---

## Dependencies

```
opencv-python
numpy
matplotlib
```

---

## Usage

```bash
python solution.py
```

Place `Shapes1.bmp` inside an `images/` directory relative to the script. Three figure windows will be displayed showing the analysis for squares, L-shapes, and the final hollow circle extraction.

---

## ⭐ ═══════════════════════════════════════════ ⭐

# Morphologische Objekterkennung mit Hit-or-Miss-Transform

## ⭐ ═══════════════════════════════════════════ ⭐

Eine Übung zur Computer Vision, die morphologische Operationen und das **Hit-or-Miss-Transform (HMT)** für formenspezifische Objekterkennung und sequentielle Pixelmanipulation mit OpenCV untersucht.

---

## Übersicht

Dieses Skript verarbeitet ein binäres Bild (`Shapes1.bmp`), das drei verschiedene Formen enthält **Quadrate**, **L-Formen** und **hohle Kreise** und isoliert jede Klasse mithilfe einer Pipeline morphologischer Operationen. Die Kernidee besteht darin, erkannte Formen schrittweise vom Originalbild zu subtrahieren, bis nur noch die Zielobjekte übrig bleiben.

---

## Pipeline

```
Original Image
      │
      ├──[HMT: square kernel]──► Square locations
      │         │
      │    [Dilate & subtract]
      │         │
      ▼         ▼
 Hollow + L-shapes
      │
      ├──[HMT: L-shape kernel]──► L-shape locations
      │         │
      │    [Reconstruct mask & subtract]
      │         │
      ▼         ▼
  Hollow Circles Only
```

---

## Wichtige Konzepte

### Binäre Darstellung

Das Graustufenbild wird geschwellenwertet und in ein binäres Bild `{0, 1}` (`binary_01`) umgewandelt, das für die Verwendung mit den morphologischen Funktionen von OpenCV erforderlich ist, die dieses Format für HMT-Operationen erwarten.

### Hit-or-Miss-Transform (HMT)

HMT ist eine morphologische Operation, die **exakte lokale Muster** in einem binären Bild erkennt. Der Kernel kodiert drei Zustände:

| Wert | Bedeutung                                       |
| ---- | ----------------------------------------------- |
| `+1` | Ein Vordergrundpixel muss vorhanden sein (Hit)  |
| `-1` | Ein Hintergrundpixel muss vorhanden sein (Miss) |
| `0`  | Egal                                            |

Ein Pixelort wird nur dann als Treffer markiert, wenn **alle** Hit-Bedingungen und **alle** Miss-Bedingungen gleichzeitig erfüllt sind. Dadurch ist HMT viel selektiver als normale Erosion oder Dilatation.

```python
cv2.morphologyEx(binary_01, cv2.MORPH_HITMISS, kernel, anchor=(r, c))
```

---

## Abschnitte

### Abschnitt 1 — Quadraterkennung und Eliminierung

Ein 6×6-Kernel mit einem soliden `+1`-Innenbereich und einem `-1`-Rand wird verwendet, um den **oben links liegenden Ankerpunkt** jedes Quadrats zu erkennen. Der Anker ist auf `(1, 1)` gesetzt, um den Innenbereich korrekt auszurichten.

```
[-1, -1, -1, -1, -1, -1]
[-1,  1,  1,  1,  1, -1]
[-1,  1,  1,  1,  1, -1]
[-1,  1,  1,  1,  1, -1]
[-1,  1,  1,  1,  1, -1]
[-1, -1, -1, -1, -1, -1]
```

Sobald die Positionen gefunden sind, wird der Pixelbereich jedes Quadrats `[r:r+4, c:c+4]` in eine Maske gemalt, die vom Original subtrahiert wird — es bleiben nur noch L-Formen und hohle Kreise übrig.

![Square Analysis](result-images/square-analysis.PNG)

---

### Abschnitt 2 — L-Form-Erkennung und Eliminierung

Ein 6×7-Kernel erkennt das **L-Form-Muster** (gefülltes Rechteck mit einer fehlenden unteren rechten Ecke). Der Anker steht auf `(0, 0)`, sodass der Erkennungspunkt der oberen linken Ecke der Form entspricht.

```
[ 1,  1,  1,  1,  1,  1,  1]
[ 1,  1,  1,  1,  1,  1,  1]
[ 1,  1,  1,  1,  1,  1,  1]
[ 1,  1,  1,  1,  1,  1,  1]
[ 1,  1,  1,  1, -1, -1, -1]
[ 1,  1,  1,  1, -1, -1, -1]
```

Anstatt die HMT-Ausgabe direkt als Maske zu verwenden (die nur einzelne Pixel markieren würde), wird die L-Form-Geometrie **rekonstruiert**, indem der richtige Pixelbereich für jede erkannte Position mit einer binären Formmaske gemalt wird. Dieser "Dilatation durch Form"-Ansatz stellt den vollständigen Objekt-Fußabdruck wieder her, bevor subtrahiert wird.

![L-shape Analysis](result-images/L-shapeAnalysis.PNG)

---

### Abschnitt 3 — Isolierung hohler Kreise

Nach den beiden Subtraktionsschritten erhält man hohle Kreise durch:

```python
hollow_only = cv2.subtract(diff_no_squares, recovered_l_shapes_255)
```

![Hollow Circles](result-images/hollowCircles.PNG)

Dies funktioniert, weil `diff_no_squares` bereits nur L-Formen und hohle Kreise enthält. Durch Subtraktion der rekonstruierten L-Form-Maske bleiben die hohlen Kreise isoliert.

---

## Beobachtungen

- **HMT ist sehr empfindlich gegenüber Kernel-Design.** Ein Kernel, der auch nur geringfügig falsch an die tatsächlichen Formabmessungen im Bild angepasst ist, liefert keine Erkennungen. Die richtige Platzierung des Ankers ist ebenso entscheidend.
- **Die HMT-Ausgabe markiert Ankerpunkte, nicht vollständige Formen.** Um erkannte Objekte zu entfernen, müssen ihre Pixelregionen explizit rekonstruiert und in eine Subtraktionsmaske gemalt werden.
- **Sequentielle Subtraktion** ist eine effektive Strategie, wenn die Formen nicht überlappen und ihre Kernel gut definiert sind.
- `cv2.subtract` behandelt das Clipping automatisch (keine negativen Werte), was es sicher für maskenbasierte Eliminierung in `uint8`-Bildern macht.

---

## Abhängigkeiten

```
opencv-python
numpy
matplotlib
```

---

## Verwendung

```bash
python solution.py
```

Platziere `Shapes1.bmp` in einem `images/`-Verzeichnis relativ zum Skript. Drei Fenster werden angezeigt, die die Analyse für Quadrate, L-Formen und die finale Isolierung hohler Kreise zeigen.
