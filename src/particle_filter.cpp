/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <stdio.h>

#include "particle_filter.h"

using namespace std;

default_random_engine random_engine;

int INFO = 10;
int DBG = 5;
int CRITICAL = 1;
int LOG_LEVEL = 0;

#define LOG(w, x)                                                       \
  {                                                                     \
    if (w <= LOG_LEVEL)                                                 \
    {                                                                   \
      cout << __FUNCTION__ << "::" << __LINE__ << "->" << x << endl;    \
    }                                                                   \
  }

void
ParticleFilter::init(
  double x,
  double y,
  double theta,
  double std[])
{
  LOG(INFO, "Enter");

  num_particles = 10;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  for (int i = 0; i < num_particles; i++)
  {
    Particle p(i,
               dist_x(random_engine),
               dist_y(random_engine),
               dist_theta(random_engine),
               1.0 / num_particles);

    particles.push_back(p);
    weights.push_back(1.0 / num_particles);
  }

  is_initialized = true;

  LOG(INFO, "Exit");
}

void
ParticleFilter::prediction(
  double delta_t,
  double std_pos[],
  double velocity,
  double yaw_rate)
{
  LOG(INFO, "Enter");

  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++)
  {
    double temp_x;
    double temp_y;
    double temp_theta;

    if (abs(yaw_rate) < 0.00001)
    {
      temp_x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
      temp_y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
    }
    else
    {
      temp_theta = particles[i].theta + yaw_rate * delta_t;
      temp_x = particles[i].x + velocity * (sin(temp_theta) - sin(particles[i].theta)) / yaw_rate;
      temp_y = particles[i].y + velocity * (cos(particles[i].theta) - cos(temp_theta)) / yaw_rate;
    }

    particles[i].x = temp_x + dist_x(random_engine);
    particles[i].y = temp_y + dist_y(random_engine);
    particles[i].theta = temp_theta + dist_theta(random_engine);
  }

  LOG(INFO, "Exit");
}

void
ParticleFilter::dataAssociation(
  std::vector<LandmarkObs> predicted,
  std::vector<LandmarkObs>& observations)
{
  LOG(INFO, "Enter");

  for (auto& obs: observations)
  {
    double min_distance = numeric_limits<double>::max();
    
    for (auto& pred: predicted)
    {
      double current_distance = dist(obs.x, obs.y, pred.x, pred.y);
      if (current_distance < min_distance)
      {
        obs.id = pred.id;
        min_distance = current_distance;
      }
    }
  }

  LOG(INFO, "Exit");
}

void
rangeFilterLandmarks(
  Particle &particle,
  double range,
  const Map &map,
  vector<LandmarkObs> &filtered_landmarks)
{
  LOG(INFO, "Enter");

  for (auto const& landmark : map.landmark_list)
  {
    // both particle and landmark must be in global co-ordinate system
    if (dist(particle.x, particle.y, landmark.x_f, landmark.y_f) <= range)
    {
      LandmarkObs l;
      l.id = landmark.id_i;
      l.x = landmark.x_f;
      l.y = landmark.y_f;

      filtered_landmarks.push_back(l);
    }
  }

  LOG(INFO, "Exit");
}

void
transformToMap(
  Particle &particle,
  const vector<LandmarkObs> &observations,
  vector<LandmarkObs> &map_observations)
{
  LOG(INFO, "Enter");

  for (auto &observation: observations)
  {
    LandmarkObs l;
    l.id = observation.id;
    l.x = particle.x + cos(particle.theta) * observation.x - sin(particle.theta) * observation.y;
    l.y = particle.y + sin(particle.theta) * observation.x + cos(particle.theta) * observation.y;

    map_observations.push_back(l);
  }

  LOG(INFO, "Exit");
}

void
ParticleFilter::updateWeights(
  double sensor_range,
  double std_landmark[],
  const std::vector<LandmarkObs> &observations,
  const Map &map_landmarks)
{
  LOG(INFO, "Enter");

  double std_x = std_landmark[0];
  double std_y = std_landmark[1];

  double weight_sum = 0;

  for (int i = 0; i < num_particles; i++)
  {
    LOG(INFO, "Particle " + to_string(i));

    vector<LandmarkObs> filtered_landmarks;
    rangeFilterLandmarks(
      particles[i],
      sensor_range,
      map_landmarks,
      filtered_landmarks);

    vector<LandmarkObs> map_observations;
    transformToMap(
      particles[i],
      observations,
      map_observations);

    // find landmark closest to observation and update particle weight
    particles[i].weight = 1;
    LOG(INFO, "#filtered_landmarks " + to_string(filtered_landmarks.size()) + ", #map_observations " + to_string(map_observations.size()));

    int j = 0;
    for (auto& obs: map_observations)
    {
      double min_distance = numeric_limits<double>::max();
      LandmarkObs closest_landmark;

      int k = 0;
      for (auto& pred: filtered_landmarks)
      {
        double current_distance = dist(obs.x, obs.y, pred.x, pred.y);
        if (current_distance < min_distance)
        {
          closest_landmark = pred;
          min_distance = current_distance;
          LOG(INFO, "Found closest; observation = " + to_string(j) + ", landmark = " + to_string(k));
          k++;
        }
      }

      double e1 = -1 * (obs.x - closest_landmark.x) * (obs.x - closest_landmark.x) / (2 * std_x * std_x);
      double e2 = -1 * (obs.y - closest_landmark.y) * (obs.y - closest_landmark.y) / (2 * std_y * std_y);
      double div = 2 * M_PI * std_x * std_y;
      LOG(INFO, "Weight update " + to_string(exp(e1 + e2) / div));
      particles[i].weight *= exp(e1 + e2) / div;
      j++;
    }

    weight_sum += particles[i].weight;
    LOG(INFO, "Particle = " + to_string(i) + ", non-normalized weight = " + to_string(particles[i].weight));
    if (INFO <= LOG_LEVEL)
    {
      LOG(INFO, "Enter any key...");
      getchar();
    }
  }

  // normalize weights
  for (int i = 0; i < num_particles; i++)
  {
    particles[i].weight /= weight_sum;
    weights[i] = particles[i].weight;
    LOG(INFO, "Particle = " + to_string(i) + ", normalized weight = " + to_string(particles[i].weight));
  }

  if (INFO <= LOG_LEVEL)
  {
    LOG(INFO, "Enter any key...");
    getchar();
  }

  LOG(INFO, "Exit");
}

void
ParticleFilter::resample()
{
  vector<Particle> old_particles;
  for (auto &p: particles)
  {
    old_particles.push_back(p);
  }

  particles.clear();

  discrete_distribution<int> distribution(weights.begin(), weights.end());
  for (int i = 0; i < num_particles; i++)
  {
    particles.push_back(old_particles[distribution(random_engine)]);
  }
}

Particle
ParticleFilter::SetAssociations(
  Particle& particle,
  const std::vector<int>& associations,
  const std::vector<double>& sense_x,
  const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string
ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string
ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

string
ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
