// The input data 
data {
 int<lower=1> n;
 array[n] int h;
 array[n] real memory; 
 array[n] real confidence;
 real prior_mean_bias;
 real<lower=0> prior_sd_bias;
 real prior_mean_beta;
 real<lower=0> prior_sd_beta;
 real prior_mean_conf;
 real<lower=0> prior_sd_conf;
 
}

// The parameters accepted by the model. 
parameters {
  real bias;
  real beta;
  real confidencerate;
}

// The model
model {
  // priors
  target += normal_lpdf(bias | prior_mean_bias, prior_sd_bias);
  target += normal_lpdf(beta | prior_mean_beta, prior_sd_beta);
  target += normal_lpdf(confidencerate | prior_mean_conf, prior_sd_conf);
  
  // model
  for (trial in 1:n) {
    target += bernoulli_logit_lpmf(h[n] | bias + (beta + confidencerate) * logit(memory[trial]));
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
  conf_posterior = confidencerate;
  
  for (i in 1:n){
    prior_choice[i] = binomial_rng(1, inv_logit(bias_prior + (beta_prior + conf_prior) * logit(memory[i])));
    posterior_choice[i] = binomial_rng(1, inv_logit(bias_posterior + (beta_posterior + conf_posterior) * logit(memory[i])));
  }
  
  prior_preds = sum(prior_choice);
  posterior_preds = sum(posterior_choice);
  
}
