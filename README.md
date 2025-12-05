# Data4Sim

![Data4Sim](dara.png)


## Abstract

Understanding human movement in industrial environments requires more than 1
simple step counts—it demands contextual information to interpret activities and enhance 2
workflows. Key factors such as location and process context are essential. However, 3
research on context-sensitive human activity recognition is limited by the lack of publicly 4
available datasets that include both human movement and contextual labels. Our work 5
introduces the DaRA dataset to address this research gap. DaRA comprises over 109 6
hours of video footage, including 32 hours from wearable first-person cameras and 77 7
hours from fixed third-person cameras. In a laboratory environment replicating a realistic 8
warehouse, scenarios such as order-picking, packaging, unpacking, and storage were 9
captured. The movements of 18 subjects were captured using inertial measurement units, 10
Bluetooth devices for indoor localization, wearable first-person cameras, and fixed third- 11
person cameras. DaRA offers detailed annotations with 12 class categories and 207 class 12
labels covering human movements and contextual information such as process steps and 13
locations. 15 annotators and 8 revisers contributed over 1572 hours in annotation and 14
361 hours in revision. High label quality is reflected in Light’s Kappa values ranging from 15
78.27% to 99.88%. Therefore, DaRA provides a solid and multimodal foundation for human 16
activity recognition and human context recognition in industrial settings.



# DaRA Annotation Preprocessing Tool

## 📘 Overview

This Python tool preprocesses the **DaRA dataset annotations** into clean, synchronized CSV files ready for analysis or machine learning.
It automatically detects available annotation types, applies the correct scheme mappings, and lets you interactively choose which data categories to include or combine.

The tool supports both **direct folders** and a single **`raw_data.zip`** archive placed next to the script.

---

## 🧩 Key Features

* **Automatic discovery** of all available annotation folders (`Final__Annotation__CCxx_*`).
* **Scheme-aware label extraction:**

  * Interprets **`CLxxx` codes**: these are class codes defined in the scheme JSON (e.g., `CL030|Upwards`) that the tool maps to human-readable labels.
* **Structured category handling:**

  * **Location annotations** (`Human` and `Cart`) automatically split into:

    * `Location – Human (Main)` / `Location – Human (Sub)`
    * `Location – Cart (Main)` / `Location – Cart (Sub)`
  * **Hand annotations** (`Left` and `Right`) split into:

    * `Primary Position`, `Type of Movement`, `Object`, and `Tool`.
* **Interactive combination builder (what “combination” means):**

  * A **combination** is a **single output column** (`input` or `output`) created by **joining** the values of the columns (and sub-parts) you select, **in the order you select** them (e.g., `Main Activity / Left Hand (Object) / Location – Human (Main)`).
  * Choose which annotations to include.
  * When a **Location** or **Hand** annotation is selected, choose which sub-parts to include (e.g., Location **Main/Sub**; Hand **Primary/Movement/Object/Tool**).
  * Optionally build:

    * `input` → (activities + sub-activities + location + IT + order)
    * `output` → (high/mid/low-level process)
* **Optional filtering** of “Unknown” and “Other” labels.
* **Saves final outputs** to `data_preprocessed/SXX.csv`.

Here’s the improved version of that section — rewritten for clarity and precision, reflecting what actually appears in your output table (as seen in the screenshot):

---

> **Note on subcolumns:**
> Subcolumns are **added automatically only when the corresponding category is selected**.
> For example:
>
> * If you select **Location – Human**, the following columns will appear:
>   `Location – Human`, `Location – Human (Main)`, and `Location – Human (Sub)`
> * If you select **Location – Cart**, you’ll get:
>   `Location – Cart`, `Location – Cart (Main)`, and `Location – Cart (Sub)`
> * If you include **Left Hand** or **Right Hand**, they expand into:
>   `Sub-Activity – (Left or Right) Hand`, `Sub-Activity – (Left or Right) Hand (Primary Position)`, `Sub-Activity – (Left or Right) Hand (Type of Movement)`, `Sub-Activity – (Left or Right) Hand (Object)`, and `Sub-Activity – (Left or Right) Hand (Tool)`
>   These subcolumns are added even if they’re not used in the combined input/output columns, ensuring all structural parts are available in the preprocessed output.

---

## 🧱 Folder Structure

Your working directory should contain:

```
preprocessing.py
raw_data.zip   ← the dataset archive (required)
```

Inside `raw_data.zip`, include:

```
raw_data/
  Final__Annotation__CC01_Main Activity/
  Final__Annotation__CC02_Sub-Activity - Legs/
  Final__Annotation__CC03_Sub-Activity - Torso/
  Final__Annotation__CC04_Sub-Activity - Left Hand/
  Final__Annotation__CC05_Sub-Activity - Right Hand/
  Final__Annotation__CC06_Order/
  ...
  scheme/
    scheme__CC01_Main Activity.json
    scheme__CC04_Sub-Activity - Left Hand.json
    scheme__CC05_Sub-Activity - Right Hand.json
    scheme__CC11_Location - Human.json
    scheme__CC12_Location - Cart.json
```

The script automatically unzips `raw_data.zip` into `raw_data/`.

---

## ⚙️ Installation

1. Ensure you have **Python 3.8+** installed.
2. (Optional) Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   ```
3. Install dependencies:

   ```bash
   pip install pandas numpy
   ```

---

## ▶️ How to Run

Simply run:

```bash
python preprocessing.py
```

### The tool will:

1. Check if a `raw_data/` folder exists.

   * If not, it will **automatically extract** `raw_data.zip`.
2. List all available annotation categories.
3. Ask which categories to keep (you can type `all` or comma-separated numbers).
4. Ask:

   * Whether to remove “Unknown” / “Other” labels.
   * Whether to create **input/output** combinations.
5. If selected:

   * Ask which parts of **Location** (Main/Sub) or **Hand** (Primary, Movement, Object, Tool) to include.
6. Process each subject and save the outputs in:

   ```
   data_preprocessed/S01.csv
   data_preprocessed/S02.csv
   ...
   ```

---

## 💾 Output

Each output CSV includes:

* One column per selected annotation.
* Automatically added subcolumns **when that category was selected**, for:

  * `Location – Human (Main/Sub)`
  * `Location – Cart (Main/Sub)`
  * `Left/Right Hand` subcomponents.
* Optional `input` and/or `output` columns (based on user choices).
* Missing or unavailable labels filled with `"-"`.

---

## 🧠 Example Session

```
Available annotation types:
  1. Main Activity
  2. Sub-Activity – Torso
  3. Sub-Activity – Left Hand
  4. Sub-Activity – Right Hand
  5. Location – Human
  6. Location – Cart
  7. Information Technology
  8. Order
  9. High-Level Process
 10. Mid-Level Process
 11. Low-Level Process
 12. All

Enter numbers of annotation types you want (e.g., 1,2,3 or 'all'): all
Do you want to remove 'Unknown' labels? (y/n): y
Do you want to remove 'Other' labels? (y/n): n
Do you want to create INPUT combinations? (y/n): y
Select which columns to combine for the INPUT combination:
  ...
Location – Human is included. Which parts do you want to use?
  1. Main location
  2. Sub location
  3. All
→ Enter: 3
Sub-Activity – Left Hand is included. Which parts do you want to use?
  1. Primary Position
  2. Type of Movement
  3. Object
  4. Tool
  5. All
→ Enter: 3
✅ Selected parts for Sub-Activity – Left Hand: Object

Sub-Activity – Right Hand is included in the INPUT combination.
Which parts do you want to use?
 1. Primary Position
 2. Type of Movement
 3. Object
 4. Tool
 5. All
Enter numbers for parts (e.g., 1,2 or 'all'): 2
✅ Selected parts for Sub-Activity – Right Hand: Type of Movement

Do you want to create OUTPUT combinations (High/Mid/Low-level processes)? (y/n): y
Select which columns to combine for the OUTPUT combination:

Available annotation types:
 1. High-Level Process
 2. Mid-Level Process
 3. Low-Level Process
 4. All

Enter the numbers of annotation types you want (e.g., 1,2,3 or 'all'): 4

✅ Selected annotation types:
   - High-Level Process
   - Mid-Level Process
   - Low-Level Process

```

---

## 📂 Output Example

**Summary of what was created:**

```
📦 Created INPUT combination using:
   - Main Activity
   - Sub-Activity – Legs
   - Sub-Activity – Torso
   - Sub-Activity – Left Hand (Object)
   - Location – Human (Main)
   - Location – Human (Sub)
   - Location – Cart (Main)
   - Location – Cart (Sub)
   - Information Technology
   - Order

🧩 Created OUTPUT combination using:
   - High-Level Process
   - Mid-Level Process
   - Low-Level Process
```

---

## 🧾 Notes

* Folder names and scheme file names **must** follow the DaRA naming convention.

---

## 🧰 Troubleshooting

| Problem                                       | Likely Cause                           | Fix                              |
| --------------------------------------------- | -------------------------------------- | -------------------------------- |
| `No annotation folders found under raw_data/` | Missing `raw_data.zip` or wrong path   | Place the zip next to the script |
| `KeyError ... CLxxx not in index`             | Wrong CSV header / inconsistent scheme | Check the scheme and CSV match   |

---

## 🧱 Packaging for Sharing

When sending to collaborators or reviewers, zip these three together:

```
preprocessing.py
raw_data.zip
README.md
```

They can simply unzip and run:

```bash
python preprocessing.py
```

The tool will automatically extract, preprocess, and save the results.
