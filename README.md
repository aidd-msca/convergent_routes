# Improving Route Development Using Convergent Retrosynthesis Planning

## Summary
This repository contains the code for the publication _Improving Route Development Using Convergent Retrosynthesis Planning_. The project introduces a multi-step synthesis planning framework to enhance retrosynthetic planning for compound libraries, focusing on convergent routes where multiple compounds share common intermediates.

## Requirements
This repository is built on conda and poetry environments - to install carry out as follows

```python
git clone https://github.com/aidd-msca/convergent_routes.git
```

```python
cd convergent_routes

conda env create -f environment.yml
conda activate convergentroutes

poetry install
```

## How to Use

### A. Convergent Routes Curation

With a collection of reaction data, there are two steps to create a convergent routes dataset. The first, allows the processing of convergent routes. The second deduplicates any routes that may be present across multiple projects, this is only necessary if using the convergent routes for test purposes.

1. Process Reactions
This step converts a collection of reactions into a collection of convergent routes, to run:

```python
python process_reactions.py -cn config
```
The code assumes that the config.yaml file is stored in ./experiments/process_reactions , though this can be updated. For further details on the settings available in the config file see section [Config Files](#config-files).

2. Deduplicate Reactions
This allows the collation of any repeated routes across projects, and cleans up any fringe cases which may have been identified, to run:

```python
python deduplicate_cluster.py -cn config
```
The code assumes that the config.yaml file is stored in ./experiments/deduplicate_routes , though this can be updated. For further details on the settings available in the config file see section [Config Files](#config-files).

### B. Multi-step Synthesis Planning

To produce convergent routes for a selection of compounds of compound libraries, run:

```python
python main.py -cn config
```
The code assumes that the config.yaml file is stored in ./experiments/convergent_search , though this can be updated. For further details on the settings available in the config file see section [Config Files](#config-files).

We highly encourage the use of GPU inference to speed up the search process.

### C. Single-Step Model

The approach assumes that you have a trained single-step model that inputs product SMILES and outputs reactant SMILES. The fine-tuned USPTO model used in [1] is available at: [INCLUDE LINK]

## Datasets
1. Processed Reaction Data <br>
Tab separated comma file containing atom-mapped reactions and project associated to each reaction

2. Convergent Routes Dataset <br>
JSON file containing convergent routes used to validate convergent search approach

3. Convergent Routes Search <br>
JSON file containing results of convergent route search experiment

## Config Files
Aspects of note for the config files are as follows,

### Process Reactions
- **process_data**: Whether to carry out the data cleaning step, such as assigning reactants/products and canonicalizing
- **skip_cycles**: Whether to skip routes which contain cycles, this step requires further processing
- **files**: List of tsv files with reaction data
- **document_id**: Name of column with project identifiers
- **max_file**: Maximum number of routes per file, will create multiple files if the number of routes is larger
- **store_root**: File path of folder to store routes

### Deduplicate Routes
- **routes_root**: File path to folder containing identified convergent routes
- **remove_duplicates**: Whether to remove duplicated routes, this is carried out across projects
- **remove_stereoisomers**: Whether to remove stereoisomers i.e. routes where the target molecules are chemical identical but differ in structure 
- **store_root**: Path to folder to store processed routes
- **store_fp**: Name of file for processed routes

### Convergent Search
- **experiment_name**: Name of the experiment, this will ensure the correct naming of files
- **k**: Number of nodes to follow up at each iteration
- **batch_size**: Batch size of single-step model, if k < batch_size this is ignored
- **searchsettings**: General settings for the multi-step searches
- - _max_time_: Maximum time to run the multi-step search per compound
- - _max_iterations_: Maximum iterations to run the multi-step search per compound
- - _max_len_route_: Maximum route length in convergent route
- - _target_molecule_limit_: Maximum number of compounds in a compound library, if above limit, compound library is not explored
- **single_step_model**: Settings for single step model, currently only Chemformer is tested however this can be adapted to any single-step model using the ModelZoo [2] package
- **target_molecules**: List of compounds from compound library
- **target_molecules_fp**: File with compounds from multiple compound libraries
- **group_col**: Named column of compound library groupings, null if target_molecules_fp not passed
- **molecule_col**: Named column of compound library smiles, null if target_molecules_fp not passed
- **buildingblocks_fp**: Path to .tsv file with starting materials, these must be previously processed and canonicalized 
- **reference_fp**: Path to experimentally validated routes if carrying out comparison, otherwise null
- **results_fp**: Path to store results
- **store_routes**: Settings to store routes, otherwise null
- - _n_: Number of routes to store per compound library
- - _store_fp_: Path to store routes
- **individual**: Whether to process compounds individually, only used for comparison purposes
- **keep_max_len**: Whether to keep reactants at maximum route length as if they were building blocks, this is not used in [1]
- **subsample**: Number of compound libraries or compounds to subsample, otherwise null

