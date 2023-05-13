## Introduction

<!-- > This repo has been populated by an initial template to help get you started. Please
> make sure to update the content to build a great experience for community-building.

As the maintainer of this project, please make a few updates:

- Improving this README.MD file to provide a great experience
- Updating SUPPORT.MD with content about this project's support experience
- Understanding the security reporting process in SECURITY.MD
- Remove this section from the README -->

This repository contains code and data for "Accurate Measures of Vaccination and Concerns of Vaccine Holdouts from Web Search Logs" (2023) by Serina Chang, Adam Fourney, and Eric Horvitz.

## Python files
``analyze_cdc_records.py``: functions to load and analyze data from CDC.

``constants_and_utils.py``: relative paths to data, common functions for data processing and loading external data (eg, from Census).

``gnn_node_classification.py``: definitions of GNN and CNN models, functions to run and evaluate model experiments.

``graph_methods.py``: functions to construct query-url graphs from Bing logs, run personalized PageRank, and load nodes for AMT annotation.

``vaccine_intent.py``: functions for identifying vaccine intent and other intents (based on regex) in queries and URLs; also for analyzing vaccine intents (eg, topic modeling).

## Notebooks
``bing_vs_google.ipynb``: comparing normalized search trends on Bing vs Google.

``classifier_results.ipynb``: results from vaccine intent classifier.

``holdout_analyses.ipynb``: constructing ontology, analyzing news/vaccine concerns of vaccine holdouts.

``vaccine_intent_rates.ipynb``: estimating regional vaccine intent rates, comparison to CDC, demographic trends.

## Vaccine intent estimates
We release estimates of regional vaccine intent rates based on our vaccine intent classifier, with estimates corrected for non-uniform Bing coverage. See Methods M3 of the paper for details.

``state_data.csv``: for each US state and Washington, D.C., we provide its estimated vaccine intent rate (with 95% CIs).

``county_data.csv``: for each US county, we provide its population size based on the 2020 5-year American Community Survey. We provide estimated vaccine intent rates (with 95% CIs) for 3,045 counties (99.8% of the total population).

``zcta_data.csv``: for each US ZIP code tabulation area (ZCTA), we provide its population size based on the 2020 5-year American Community Survey and mapping to US county, based on largest overlap in land area. We provide estimated vaccine intent rates (with 95% CIs) for 20,899 ZCTAs (97.6% of the total population).

## Ontology
Our hierarchical ontology consists of 4 levels: URLs, URL clusters, subcategories, and top categories. We use Louvain community detection to automatically partition vaccine-related URLs into clusters, then we manually label clusters with 1-2 subcategories and organize subcategories into top categories. Here, we report click and user counts based on the clicks of holdouts and their matched early adopters from April to August 2021. See Methods M4 of the paper for details.

``ontology_subcats.csv``: maps subcategory to top category. Each row represents a subcategory, its number of URLs and clicks, and its top category.

``ontology_clusters.csv``: maps cluster to subcategory. Each row represents a cluster, its number of URLs and clicks, and its assigned 1-2 subcategories.

``ontology_urls.csv``: maps URL to cluster. Each row represents a URL, its number of clicks and number of users who clicked on it (among holdouts and early adopters), its assigned cluster, and the proportion of cluster clicks that this URL accounts for. For privacy reasons, we only include URLs that were clicked on by at least 10 users, which leaves 13,043 URLs.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
