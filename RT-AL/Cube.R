####
## Functions
####

first_sampling= function(data,K, method = 'stratifiedCube', n_sampling_cluster = 300){
  # data 
  # K: number of points to be labeled
  # n_sampling_cluster: number of points to be subsampled to run Mclust
  # method: how points are labeled
  # 'stratifiedCube' by default, with uniform weights, where strates are computed using a clustering
  # 'uniform' using a sample in each leaf
  # 'cube' using a cube in each leaf
  # 'GSx' using GsX https://arxiv.org/pdf/1808.04245.pdf
  p = ncol(data)-1
  n = nrow(data)
  if (method == 'uniform'){
    ech = sample(1:n,K)
  } else if (method == 'cube'){
    incl_prob = rep(K/n,n)
    ech = cube(incl_prob,as.matrix(cbind(data[,-(p+1)], incl_prob)))
  } else if (method == 'stratifiedCube'){
    sample_clus = sample(1:n, n_sampling_cluster)
    res_clus = Mclust(data[sample_clus,-(p+1)], G = 2:9, modelNames =  "VEI", verbose = FALSE)
    pred_clus = predict(res_clus, data[,-(p+1)])$classification
    variance_clus = res_clus$parameters$variance$scale
    incl_prob = rep(K/n,n)
    ech = cubestratified(incl_prob, as.matrix(cbind(data[, -(p+1)], incl_prob)), pred_clus)
    ech = which(ech == 1)
  }
  return(ech)
}