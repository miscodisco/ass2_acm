// The input data 
data {
 int<lower=1> n;
 array[n] int h;
 array[n] int other;
 real prior_mean_bias;
 real prior_sd_bias;
 real prior_mean_beta;
 real prior_sd_beta;
 real prior_mean_conf;
 real prior_sd_conf;
 
}

// The parameters accepted by the model. 
parameters {
  real bias;
  real beta;
  real confidencerate;
}

transformed parameters{
  vector[n] memory;

  for (trial in 1:n){
  if (trial == 1) {
    memory[trial] = 0.5;
  } 
  if (trial < n){
      memory[trial + 1] = memory[trial] + ((other[trial] - memory[trial]) / trial);
      if (memory[trial + 1] == 0){memory[trial + 1] = 0.01;}
      if (memory[trial + 1] == 1){memory[trial + 1] = 0.99;}
    }
  }
  
  vector[n] confidence;
  for (trial in 1:n){
    if (trial == 1) {
        confidence[trial] = 0;} 
    if (trial < n){
      if (other[trial] == h[trial]){
        confidence[trial+1] = confidence[trial] + confidencerate;
       }
        else {
        confidence[trial+1] = confidence[trial] - confidencerate;
      }
    }
  }
}

// The model to be estimated. 
model {
  // Priors
  target += normal_lpdf(bias | prior_mean_bias, prior_sd_bias);
  target += normal_lpdf(beta | prior_mean_beta, prior_sd_beta);
  target += normal_lpdf(confidencerate | prior_mean_conf, prior_sd_conf);
  
  // Model, looping to keep track of memory
  for (trial in 1:n) {
    target += bernoulli_logit_lpmf(h[trial] | bias + (beta + confidence[trial]) * logit(memory[trial]));
  }
}

// Saving Priors & posteriors
generated quantities{
  real bias_prior;
  real bias_posterior;
   
  real beta_prior;
  real beta_posterior;
  
  real conf_prior;
  real conf_posterior;
  
  int <lower=0, upper=n> prior_preds;
  int <lower=0, upper=n> posterior_preds;
  
  array[n] int <lower=0, upper=n> prior_choice;
  array[n] int <lower=0, upper=n> posterior_choice;
  
  bias_prior = normal_rng(0,1);
  bias_posterior = inv_logit(bias);
  
  beta_prior = normal_rng(0,1);
  beta_posterior = inv_logit(beta);
  
  conf_prior = normal_rng(0,1);
  conf_posterior = inv_logit(confidencerate);
  
  for (i in 1:n){
    prior_choice[i] = binomial_rng(1, inv_logit(bias_prior + (beta_prior + conf_prior) * memory[i]));
    posterior_choice[i] = binomial_rng(1, inv_logit(bias_posterior + (beta_posterior + conf_posterior) * memory[i]));
  }
  
  prior_preds = sum(prior_choice);
  posterior_preds = sum(posterior_choice);
  
}
