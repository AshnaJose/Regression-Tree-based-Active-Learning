# Regression-Tree-based-Active-Learning

Contains datasets and codes for active learning in regression using regression trees (RT-AL). Contains codes used in "Jose, A., de Mendonça, J.P.A., Devijver, E. et al. Regression tree-based active learning. Data Min Knowl Disc (2023). https://doi.org/10.1007/s10618-023-00951-7"

The file contains the codes used to generate the learning curves for all datasets, for passive sampling, different model-free and model-based active learning methods, and our approach.

tree.py corresponds to RT-AL using random sampling as the query criteria in the leaves

tree_diversity.py corresponds to RT-AL using diversity-based query criteria in the leaves

tree_representativity.py corresponds to RT-AL using representativity-based query criteria in the leaves

