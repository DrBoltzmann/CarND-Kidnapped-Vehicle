/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  // Define the number of particles to generate
  // 100 seemed to provide a good coverage while not impacting computation time.
  
  num_particles = 100;
  default_random_engine gen;  
  
  // Define normal Gaussian distributions for the 2D particle states
  // Partcile state defined by position and orientation values
  normal_distribution<double> N_x(x, std[0]);
  normal_distribution<double> N_y(y, std[1]);
  normal_distribution<double> N_theta(theta, std[2]);
  
  // Sample from the normal distributions
  // Defines a distribution around each particle and state information
  for (int i = 0; i < num_particles; i++){
    Particle particle;
    particle.id = i;
    particle.x = N_x(gen);
    particle.y = N_y(gen);
    particle.theta = N_theta(gen);
    particle.weight = 1.0; // Set inital particle weight to 1
    particles.push_back(particle);
  }
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  // Define distributions
  default_random_engine gen;
  
  normal_distribution<double> N_x(0, std_pos[0]);
  normal_distribution<double> N_y(0, std_pos[1]);
  normal_distribution<double> N_theta(0, std_pos[2]);
  
  // Look for very small yaw rate values
  for(int i = 0; i < num_particles; i++) {    
    if(fabs(yaw_rate) < 0.00001){
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].x += velocity * delta_t * sin(particles[i].theta);
    } else {
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;      
    }
    // Add in noise
    particles[i].x += N_x(gen);    
    particles[i].y += N_y(gen);    
    particles[i].theta += N_theta(gen);    
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  for(int i = 0; i < observations.size(); i++) {
    // Define vector sizes
    int Num_Obs = observations.size();
    int Num_Preds = predicted.size();
    
    // Iterate through observations and then
    // take prediction for each observation
    for(int i = 0; i < Num_Obs; i++) { 
      // Define large value for minimum distance
      double minDist = 1000000;
      
      // Take id of landmark from map to be associated with the observation
      int mapId = -1;
      
      // Take observation and prediction values and calculate 2D difference
      for(int j = 0; j < Num_Preds; j++ ) {
        double diff_x = observations[i].x - predicted[j].x;
        double diff_y = observations[i].y - predicted[j].y;
        double distance = pow(diff_x, 2) + pow(diff_y, 2);
        
        // Store the id when the difference is less than the minimum
        if(distance < minDist) {
          minDist = distance;
          mapId = predicted[j].id;
        }
        observations[i].id = mapId;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations, const Map &map_landmarks){
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  // Iterate through particle list and extract state (x, y, theta) values
  for(int i = 0; i < num_particles; i++) {
    double particles_x = particles[i].x;
    double particles_y = particles[i].y;
    double particles_theta = particles[i].theta;
    
    // Create a vector to hold the map landmark locations
    // that are predicted to be within sensor range of the particle
    vector<LandmarkObs> predictions;
    
    // Iterate through landmark values
    for(int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      
      // From the landmark values, extract the state (x, y) and id
      float landmark_x = map_landmarks.landmark_list[j].x_f;
      float landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;
      
      // Find the absolute value of the state difference between landmarkd and particles
      // then use only landmarks within sensor range of the particle
      float lmark_part_diff_x = landmark_x - particles_x;
      float lmark_part_diff_y = landmark_y - particles_y;
      
      if(fabs(lmark_part_diff_x) <= sensor_range && fabs(lmark_part_diff_y) <= sensor_range) {
        predictions.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
      }
    }
    
    // Create a list of observations that are transformed
    // from the vehicle coordinates to the map coordinates
    vector<LandmarkObs> trans_os;
    for(int k = 0; k < observations.size(); k++) {
      double t_x = cos(particles_theta)*observations[k].x - sin(particles_theta)*observations[k].y + particles_x;
      double t_y = sin(particles_theta)*observations[k].x + cos(particles_theta)*observations[k].y + particles_y;
      trans_os.push_back(LandmarkObs{ observations[k].id, t_x, t_y });
    }
    
    // Associate the data for the predictions and the transformed
    // and observations of the current particle
    dataAssociation(predictions, trans_os);
    particles[i].weight = 1.0; // Set weights to 1
    
    for(int j = 0; j < trans_os.size(); j++) {
      double o_x, o_y, pr_x, pr_y;
      o_x = trans_os[j].x;
      o_y = trans_os[j].y;
      
      int asso_prediction = trans_os[j].id;
      
      // State (x,y) of the prediction associated with the current observation
      for(int k = 0; k < predictions.size(); k++) {
        if(predictions[k].id == asso_prediction) {
          pr_x = predictions[k].x;
          pr_y = predictions[k].y;
        }
      }
      
      // Weight for this observation with multivariate Gaussian
      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      double obs_w = ( 1/(2*M_PI*s_x*s_y)) * exp( -( pow(pr_x-o_x,2)/(2*pow(s_x, 2)) + (pow(pr_y-o_y,2)/(2*pow(s_y, 2))) ) );
      
      // Product for current obersvation weight and the total observations weight
      particles[i].weight *= obs_w;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  default_random_engine gen;
  discrete_distribution<int> distribution(weights.begin(), weights.end());
  
  // Define vector to hold resampled particles
  std::vector<Particle> resampled_particles;

  weights.clear();
  
  // Iterate through particle list, and resample
  for(int i=0; i < num_particles; i++){
    int chosen = distribution(gen);
    resampled_particles.push_back(particles[chosen]);
    weights.push_back(particles[chosen].weight);
  }
  
  particles = resampled_particles;

  return;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  // and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  //Clear the previous associations
  
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}