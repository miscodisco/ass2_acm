// The input (data) for the model. n of trials and h for (right and left) hand
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
  real bias; // how likely is the agent to pick right when the previous rate has no information (50-50)?
  real beta; // how strongly is previous rate impacting the decision?
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

generated quantities {
  // for the prior predictive checks 
  real <lower=0, upper=1> bias_prior;
  real <lower=0, upper=1> bias_posterior;
   
  real <lower=0, upper=1> beta_prior;
  real <lower=0, upper=1> beta_posterior;
  
  // setting confidence and memory to specific values to see how the predictions change
  // memory = 0.3, and confidence set to 0.5, 0.3 and 0.9
  int prior_preds_m03_c05;
  int posterior_preds_m03_c05;
  int prior_preds_m03_c03;
  int posterior_preds_m03_c03;
  int prior_preds_m03_c09;
  int posterior_preds_m03_c09;
  
  // memory = 0.5, and confidence set to 0.5, 0.3 and 0.9
  int prior_preds_m07_c05;
  int posterior_preds_m07_c05;
  int prior_preds_m07_c03;
  int posterior_preds_m07_c03;
  int prior_preds_m07_c09;
  int posterior_preds_m07_c09;
  
  // memory = 1.3, and confidence set to 0.5, 0.3 and 0.9
  int prior_preds_m13_c05;
  int posterior_preds_m13_c05;
  int prior_preds_m13_c03;
  int posterior_preds_m13_c03;
  int prior_preds_m13_c09;
  int posterior_preds_m13_c09;
  
  // setting the prior and posterior for bias and beta for the predictive checks
  bias_prior = inv_logit(normal_rng(0,1));
  bias_posterior = inv_logit(bias);
  
  beta_prior = inv_logit(normal_rng(0,1));
  beta_posterior = inv_logit(beta);
  
  // then getting the predictions using the priors and the fixed values for confidence and memory
  prior_preds_m03_c05 = binomial_rng(n, inv_logit(bias_prior + (beta_prior+0.5) * 0.3));
  posterior_preds_m03_c05 = binomial_rng(n, inv_logit(bias_posterior + (beta_posterior+0.5) * 0.3));
  prior_preds_m03_c03 = binomial_rng(n, inv_logit(bias_prior + (beta_prior+0.3) * 0.3));
  posterior_preds_m03_c03 = binomial_rng(n, inv_logit(bias_posterior + (beta_posterior+0.3) * 0.3));
  prior_preds_m03_c09 = binomial_rng(n, inv_logit(bias_prior + (beta_prior+0.9) * 0.3));
  posterior_preds_m03_c09 = binomial_rng(n, inv_logit(bias_posterior + (beta_posterior+0.9) * 0.3));
  
  prior_preds_m07_c05 = binomial_rng(n, inv_logit(bias_prior + (beta_prior+0.5) * 0.7));
  posterior_preds_m07_c05 = binomial_rng(n, inv_logit(bias_posterior + (beta_posterior+0.5) * 0.7));
  prior_preds_m07_c03 = binomial_rng(n, inv_logit(bias_prior + (beta_prior+0.5) * 0.7));
  posterior_preds_m07_c03 = binomial_rng(n, inv_logit(bias_posterior + (beta_posterior+0.5) * 0.7));
  prior_preds_m07_c09 = binomial_rng(n, inv_logit(bias_prior + (beta_prior+0.5) * 0.7));
  posterior_preds_m07_c09 = binomial_rng(n, inv_logit(bias_posterior + (beta_posterior+0.5) * 0.7));
  
  prior_preds_m13_c05 = binomial_rng(n, inv_logit(bias_prior + (beta_prior+0.5) * 1.3));
  posterior_preds_m13_c05 = binomial_rng(n, inv_logit(bias_posterior + (beta_posterior+0.5) * 1.3));
  prior_preds_m13_c03 = binomial_rng(n, inv_logit(bias_prior + (beta_prior+0.3) * 1.3));
  posterior_preds_m13_c03 = binomial_rng(n, inv_logit(bias_posterior + (beta_posterior+0.3) * 1.3));
  prior_preds_m13_c09 = binomial_rng(n, inv_logit(bias_prior + (beta_prior+0.9) * 1.3));
  posterior_preds_m13_c09 = binomial_rng(n, inv_logit(bias_posterior + (beta_posterior+0.9) * 1.3)); 
}
