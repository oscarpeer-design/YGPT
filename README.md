# YGPT

YGPT is a command-line tool that reviews code, writes documentation and answers questions about a codebase. It is built around a **small language model that runs entirely on your own computer** (under 1 GB), so it works on modest hardware, needs no network connection while it runs, and sends no code anywhere.

It works on **Python** (`.py`) and **C++** (`.cpp`, `.h`, `.hpp`) code.

|Command|What it does|
|-|-|
|`REVIEW`|Looks for performance and security problems in a file or a whole repository.|
|`DOCUMENT`|Writes a markdown description of a file or a whole repository.|
|any other text|A `CHAT` question, answered from the code that `REVIEW` last read.|

## How it works, in short

Plain code does most of the work: it splits the code into functions, finds the risky-looking lines with pattern searches, works out which functions call which, and finds the files that depend on each other. The small model is then asked one narrow question at a time (for example "is this flagged line really a problem?" or "what does this function do?") and its answer is put in the report.

This means the report has two kinds of statements:

* **Found by code** (reliable): which files depend on which, which functions call a function, what a change could affect, why a line was flagged, where a function is called inside a loop.
* **Written by the small model** (has to be checked): the sentences saying what a function does and the problem and fix given for a flagged line. Every report says so at the end.

## What you need

|Requirement|Notes|
|-|-|
|**Python 3.10 or newer**|Developed on 3.13.|
|**Python packages**|`faiss-cpu`, `numpy`, `onnxruntime`, `tokenizers`, `tqdm`, `tree-sitter`, `tree-sitter-cpp`, `tree-sitter-python`. `psutil` is optional.|
|**llama.cpp**|Provides the `llama-cli` program, which runs the language model. It must be on your `PATH`.|
|**A language model** (`.gguf` file, under 1 GB)|`qwen2.5-coder-0.5b-instruct-q4\_k\_m.gguf` is recommended and was used for all development.|
|**Two embedding model files**|`model.onnx` and `tokenizer.json` (the all-MiniLM-L6-v2 model), kept in the same folder as `YitzGPT.py`.|

The embedding model is a second, much smaller model that turns code into numbers so that `CHAT` and the review can find related code. It runs through `onnxruntime`, so **PyTorch and `sentence-transformers` are not needed**, and nothing is downloaded when the program runs.

## Setup

1. **Get the code.** Clone or copy the repository, and open a terminal in its folder. Always run YitzGPT from this folder (see "Notes" below).
2. **Install the Python packages.**

```
   python -m pip install faiss-cpu numpy onnxruntime tokenizers tqdm tree-sitter tree-sitter-cpp tree-sitter-python
   ```

   Optionally add `psutil`. It lets the program count physical CPU cores; without it the program uses the logical core count.

3. **Install llama.cpp.** Download a release (or build it) from [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) and make sure that typing `llama-cli --help` in a new terminal works. If it does not, add its folder to your `PATH`.
4. **Download a language model.** Get a `.gguf` file under 1 GB. The recommended one is **Qwen2.5-Coder 0.5B Instruct, quantised `q4\_k\_m`**, from Hugging Face. A larger model is refused: the limit is `MAX\_MODEL\_SIZE` in `config.py`.
5. **Tell YitzGPT where the language model is.** Create a text file called `model\_path.txt` in the repository folder. Its **first line** must be the absolute path of the `.gguf` file, for example:

```
   C:\\Models\\qwen2.5-coder-0.5b-instruct-q4\_k\_m.gguf
   ```

6. **Add the embedding model files.** From the `sentence-transformers/all-MiniLM-L6-v2` page on Hugging Face, download:

   * `onnx/model.onnx` (the full-precision file, about 90 MB), and
   * `tokenizer.json` (from the top level of the same page).

   Save both in the repository folder, next to `YitzGPT.py`, with exactly these names. (If you have already used `sentence-transformers` on this machine, a copy of `tokenizer.json` is in your Hugging Face cache folder.)

7. **Check the setup.**

```
   python package\_info.py
   ```

   It reads the imports of every Python file in the repository and reports each package as `OK`, `MISSING` or `BROKEN` (installed but cannot be loaded), with its version. For anything missing it prints the exact `pip install` line to run. It exits with code 0 when everything needed is present and 1 otherwise, so a script or pipeline can use it. It cannot check `llama-cli`, the language model, or the two embedding files: the program tells you at start-up if the language model is wrong, and at the first `REVIEW` or `CHAT` if an embedding file is missing.

## Running it

```
python YitzGPT.py
```

The program checks the language model, prints a short help text and waits at `Enter Query Type:`. Type a command, press Enter, and when it finishes you are asked for the next one.

* Commands `REVIEW` and `DOCUMENT` can be typed in any case. `HELP` and `EXIT` must be typed in **capital letters**.
* Paths must be **absolute**. Put a path in quotes if it contains spaces.
* Press **Ctrl+C** during a `REVIEW` or `DOCUMENT` to stop early. What has been done so far is still reported.

### REVIEW

```
REVIEW <file or folder> \[report file]
```

Looks for performance and security problems.

```
REVIEW C:\\Projects\\MyApp
REVIEW C:\\Projects\\MyApp\\src\\parser.cpp C:\\Reports\\parser\_review.md
```

* With a **file**, it reviews that file. With a **folder**, it reviews every supported code file in it, including subfolders.
* Without a report file, the findings are printed to the screen as they are found. With one, a markdown report is written as well. The report file must end in `.md` or `.txt`; missing folders are created.

How it works:

1. The code is split into functions and classes.
2. Pattern searches flag risky lines (for example unsafe casts, allocations sized by data, loops inside loops, a container that grows without `reserve`, a string built by repeated `+=`).
3. The flagged functions are ranked. Functions called inside a loop elsewhere, and functions that are hard to follow, come first.
4. For each of the most worthwhile ones, the model is shown the function and asked whether the flagged line is a real problem. A claim about a line that was not flagged, a claim that argues against itself, or one that only repeats the flag is thrown away and counted.
5. A report is produced with the findings, why each line was looked at, and which functions were too large to review as a whole.

Reviewing also **saves the embeddings of the code** (see CHAT). That is what makes the review able to look up related code, and what lets you ask questions afterwards.

### DOCUMENT

```
DOCUMENT <file or folder> \[markdown file]
```

Writes a description of the code.

```
DOCUMENT C:\\Projects\\MyApp C:\\Docs\\MyApp.md
```

For a repository the document contains an overview, a diagram of which file uses which, and one section per file with what the file is for, what its important functions do (the six most important per file), what a change could affect, and everything the file defines. The model writes only the plain-language sentences; the dependencies, callers and impact lists are worked out by code.

Without a markdown file, the document is printed to the screen. The markdown file must end in `.md` or `.txt`.

### CHAT

Type anything that does not start with `REVIEW` or `DOCUMENT`:

```
how does the retriever find related chunks
```

The question is matched to the most similar pieces of code from the **last repository or file you ran `REVIEW` on**, the model answers from that code, and the sources (file, lines, function, similarity) are listed under the answer. Run `REVIEW` on your code first. If nothing similar enough is found, the program says so and the model answers without code, so treat that answer with care.

Each question stands alone: CHAT has no memory of the previous question, so write follow-ups so that they make sense by themselves.

### HELP and EXIT

`HELP` shows the instructions again. `EXIT` leaves the program.

## Settings

Everything adjustable is in `config.py`, and each setting has a comment. The ones most often changed:

|Setting|What it controls|
|-|-|
|`MAX\_MODEL\_SIZE`|Largest language model file accepted (1 GB).|
|`MODEL\_CTX`|How much text the language model can read at once (4096 tokens).|
|`REVIEW\_TIME\_BUDGET\_SECONDS`, `REVIEW\_REPOSITORY\_TIME\_BUDGET\_SECONDS`|How long the model may take for a file (300 s) or a repository (600 s).|
|`DOCUMENT\_TIME\_BUDGET\_SECONDS`, `DOCUMENT\_REPOSITORY\_TIME\_BUDGET\_SECONDS`|The same for documentation. Raise these if the report says that parts were "not written".|
|`DOCUMENT\_OVERVIEW\_RESERVE\_SECONDS`|Time each file keeps back for its overview.|
|`DOCUMENT\_FUNCTIONS\_PER\_FILE`|How many functions per file are described one by one (6).|
|`CHAT\_SIMILARITY\_THRESHOLD`|How similar code must be to a question to be used (0 to 1). Higher means fewer but better matches.|
|`CHAT\_MAX\_ANSWER\_TOKENS`|The most the model may write for a CHAT answer.|
|`EMBEDDING\_MAX\_TOKENS`|How much of each code chunk the embedding model reads (256 tokens).|

The time settings are a guideline for the model's time only. A run stops starting new model calls when its share of the time is used up, and the report says what was left undone.

## Files and folders

|Path|Purpose|
|-|-|
|`YitzGPT.py`|The program: reads commands and runs them.|
|`config.py`|All settings and prompts.|
|`model\_path.txt`|You create this: the path of the language model.|
|`model.onnx`, `tokenizer.json`|The embedding model.|
|`.embedding\_cache/`|Saved embeddings (one `.npz` file per version of the code). Safe to delete at any time; they are rebuilt by the next `REVIEW`. Old files are not removed automatically.|
|`package\_info.py`|Checks that the needed Python packages are installed.|
|other `.py` files|Parsing, ranking, review, documentation and retrieval code.|

## Notes and limits

* **Run it from the repository folder.** `model\_path.txt` is looked up in the folder you start the program from. The embedding files and the cache are found next to the code.
* **Check what the model writes.** The model is very small. Flags (where and why a line was looked at) are usually worth a glance; the problem and fix sentences are often generic or wrong, and a suggested fix should never be applied without reading the code.
* **C++ review is the weakest part.** The pattern searches are rough, especially for heavily templated or macro-heavy code, and some patterns match harmless code.
* **Name-based call graph.** Callers are matched by function name, so a very common name (such as `size`) can be credited with calls that belong to another function of the same name.
* **Only Python and C++** files are read. Other files in a folder are ignored.
* **Long chunks are embedded by their first 256 tokens only**, which can make `CHAT` miss the end of a very long function.
* **Large embedding file in git.** `model.onnx` is about 90 MB. Commit it once and do not replace it, since every version stays in the repository history.

