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

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    num_particles = 1000;

    std::default_random_engine gen;
    std::normal_distribution<double> x_position(x, std[0]);
    std::normal_distribution<double> y_position(y, std[1]);
    std::normal_distribution<double> theta_position(theta, std[2]);

    for (int i = 0; i < num_particles; i++) {
        Particle p;
        p.id = i;
        p.x = x_position(gen);
        p.y = y_position(gen);
        p.theta = theta_position(gen);
        p.weight = 1;

        particles.push_back(p);
        weights.push_back(p.weight);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    std::default_random_engine gen;
    std::normal_distribution<double> x_noise(0, std_pos[0]);
    std::normal_distribution<double> y_noise(0, std_pos[1]);
    std::normal_distribution<double> t_noise(0, std_pos[2]);

    double yaw_dt = yaw_rate * delta_t;

    for (int i = 0; i < particles.size(); i++) {
        if (fabs(yaw_rate) < 0.00001) {
            particles[i].x +=  velocity * std::cos(particles[i].theta * delta_t);
            particles[i].y +=  velocity * std::sin(particles[i].theta * delta_t);
        }
        else {
            particles[i].x +=  (velocity / yaw_rate) * (std::sin(particles[i].theta + yaw_dt) - std::sin(particles[i].theta));
            particles[i].y +=  (velocity / yaw_rate) * (-std::cos(particles[i].theta + yaw_dt) + std::cos(particles[i].theta));
        }

        particles[i].x += x_noise(gen);
        particles[i].y += y_noise(gen);
        particles[i].theta += yaw_dt + t_noise(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
    double multigaus_coef = (1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]));

    for (int p = 0; p < particles.size(); p++) {
        double px = particles[p].x;
        double py = particles[p].y;
        double pt = particles[p].theta;
        double particle_weight = 1.0;

        // process each observation in respect to given particle
        for (int l = 0; l < observations.size(); l ++) {
            double ox = observations[l].x;
            double oy = observations[l].y;

            // transform from vehicle (observation) to global (map) coordinates
            double transx = px + ox * std::cos(pt) - oy * std::sin(pt);
            double transy = py + ox * std::sin(pt) + oy * std::cos(pt);

            // associate observation with nearest map landmark
            double nearest_x, nearest_y;
            double nearest_distance = sensor_range;
            for (int m = 0; m < map_landmarks.landmark_list.size(); m++) {
                double mx = map_landmarks.landmark_list[m].x_f;
                double my = map_landmarks.landmark_list[m].y_f;
                double distance = dist(transx, transy, mx, my);

                if (distance < nearest_distance) {
                    nearest_x = mx;
                    nearest_y = my;
                    nearest_distance = distance;
                }
            }

            // calculate multivariate probability
            double p_numerator_1 = ((transx - nearest_x) * (transx - nearest_x)) / (2 * std_landmark[0] * std_landmark[0]);
            double p_numerator_2 = ((transy - nearest_y) * (transy - nearest_y)) / (2 * std_landmark[1] * std_landmark[1]);
            double p_obs = multigaus_coef * std::exp(- (p_numerator_1 + p_numerator_2));
            particle_weight *= p_obs;
        }

        particles[p].weight = particle_weight;
        weights[p] = particle_weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::default_random_engine gen;
    std::discrete_distribution<> disc(weights.begin(), weights.end());

    std::vector<Particle> newParticles;
    for (int p = 0; p < num_particles; p ++) {
        int surviviorIndex = disc(gen);
        newParticles.push_back(particles[surviviorIndex]);
    }

    particles = newParticles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
