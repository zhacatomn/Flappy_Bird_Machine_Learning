#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include <bits/stdc++.h>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

// initialisers
sf::RenderWindow window(sf::VideoMode(1500, 1000), "Flappy Bird");
sf::Vector2u size = window.getSize();
int width = size.x;
int height = size.y;
int Layers[3] = {2, 200, 1};

#include "GANN.cpp"


class Bird{
public:
	NN n;
	sf::RectangleShape ent;
	float fitness, prev_fitness;
	int velocityY;
	bool dead;
	Bird(){
		ent.setSize(sf::Vector2f(75, 75));
		ent.setFillColor(sf::Color::Yellow);
		ent.setPosition(200, (height - ent.getSize().y) / 2);
		dead = false;
		velocityY = 0;
		fitness = 0;
		prev_fitness = 0;
	}

};

bool wts1(Bird* a, Bird* b){
	return (a -> prev_fitness > b -> prev_fitness);
}

bool wts2(Bird* a, Bird* b){
	return (a -> fitness > b -> fitness);
}

deque<Bird*>birds;

int main(int argc, char const *argv[])
{

	// creating the score counter
	sf::Font font;
	font.loadFromFile("manaspc.ttf");
	sf::Text scoreboard;
	scoreboard.setFillColor(sf::Color::White);
	scoreboard.setFont(font);
	scoreboard.setCharacterSize(50);
	//scoreboard.setPosition((width - scoreboard.getLocalBounds().width) / 2, 100);

	// creating a marker
	sf::RectangleShape marker;
	marker.setSize(sf::Vector2f(25, 25));
	marker.setFillColor(sf::Color::Blue);
	marker.setPosition(0, 0);

	// creating the bird
	for(int i = 0; i < 10; i++){
		birds.push_back(new Bird);
	}

	std::deque <sf::RectangleShape*> pipes;

	// VARIABLES
	int generation = 0, alive = 0;
	int velocityX = -5, distance = 0;
	int openingY = 0;
	int score = 0, landmark = 0, hiscore = 0;
	int nextPipeIndex = 0;
	const int sizeOfGap = 350;
	bool gameStart = true, gameRestart = false;

	window.setFramerateLimit(60);

	// randomize seed
	srand (time(NULL));

	while(window.isOpen()){
		sf::Event event;

		while (window.pollEvent(event)){
			switch (event.type){
			// window closed
			case sf::Event::Closed:
				window.close();
				break;

			case sf::Event::KeyPressed:
				/*
				if(event.key.code == sf::Keyboard::Space and gameStart){
					velocityY = -15;
				}
				*/

				if(event.key.code == sf::Keyboard::C){
					window.close();
				}
				break;

			// we don't process other types of events
			default:
				break;
			}
		}

		if(!gameStart)continue;
		if(gameRestart){
			/*
			window.clear(sf::Color::Black);

			window.draw(bird);
			for(int i = 0; i < pipes.size(); i++){
				pipes[i] -> move(-velocityX, 0);
				velocityX--;
				window.draw(*pipes[i]);
			}

			window.draw(scoreboard);

			window.display();
			*/
			//if(pipes[0] -> getPosition().x >= width){
			velocityX = -5;
			score = 0; landmark = 0;
			nextPipeIndex = 0;
			pipes.clear();

			sort(birds.begin(), birds.end(), wts1);
			stable_sort(birds.begin(), birds.end(), wts2);

			for(int i = 0; i < birds.size(); i++){
				birds[i] -> prev_fitness = birds[i] -> fitness;
				birds[i] -> ent.setPosition(200, (height - birds[i] -> ent.getSize().y) / 2);
				birds[i] -> velocityY = 0;
				birds[i] -> dead = false;
			}

			// remove the last 6 (survivial of the fittest)
			for(int i = 0; i < 6; i++){
				birds.pop_back();
			}

			// generate children
			for(int i = 0; i < 6; i++){
				birds.push_back(new Bird);
				// randomise parent
				int firstParent = rand() % 4;
				int secondParent = 0;
				do{
					secondParent = rand() % 4;
				}while(firstParent == secondParent);

				if(i <= 1){
					firstParent = 0;
					secondParent = 1;
				}

				// randomise cutoff point
				int cutOffPoint = rand() % Layers[1];
				while(cutOffPoint == 0){
					cutOffPoint = rand() % Layers[1];
				}

				// adjust weights on first layer
				ArrayXXf arrayW1 = birds.back() -> n.weights[0].array();
				ArrayXXf arrayW2(Layers[1], Layers[0]);

				for(int k = 0; k < Layers[1]; k++){
					if(k >= cutOffPoint){
						arrayW2 = birds[secondParent] -> n.weights[0].array();
					}
					else{
						arrayW2 = birds[firstParent] -> n.weights[0].array();
					}

					for(int z = 0; z < Layers[0]; z++){
						arrayW1(k, z) = arrayW2(k, z);
						int mutation = rand() % 100;
						if(mutation < 20 and i > 1)
							arrayW1(k, z) = ((rand() % 2) - 1) * (1 / sqrt(Layers[1]));
					}

				}

				birds.back() -> n.weights[0] = arrayW1.matrix();

				// Adjust weights on second layer
				ArrayXXf arrayW3 = birds.back() -> n.weights[1].array();

				ArrayXXf arrayW4(Layers[2], Layers[1]);
				if(cutOffPoint >= Layers[1] / 2)
					arrayW4 = birds[secondParent] -> n.weights[1].array();
				else
					arrayW4 = birds[firstParent] -> n.weights[1].array();

				for(int k = 0; k < Layers[1]; k++){
					arrayW3(0, k) = arrayW4(0, k);
				}

				birds.back() -> n.weights[1] = arrayW3.matrix();

			}

			distance = 0;

			generation++;

			gameRestart = false;
			//}
			continue;
		}

		// creating the pipes
		if(pipes.size() == 0 or width - pipes.back() -> getPosition().x - pipes.back() -> getSize().x >= 400){

			openingY = rand() % (height - sizeOfGap + 10);

			pipes.push_back(new sf::RectangleShape(sf::Vector2f(250, openingY)));
			pipes.back() -> setPosition(width, 0);
			pipes.back() -> setFillColor(sf::Color::Green);

			pipes.push_back(new sf::RectangleShape(sf::Vector2f(250, height - openingY - sizeOfGap)));
			pipes.back() -> setPosition(width, openingY + sizeOfGap);
			pipes.back() -> setFillColor(sf::Color::Green);
		}

		// gravity
		for(int i = 0; i < birds.size(); i++){
			if(birds[i] -> dead)continue;

			alive++;

			birds[i] -> velocityY++;

			ArrayXXf input(2, 1);
			ArrayXXf result(1, 1);
			// vertical distance to gap
			input(0, 0) = pipes[nextPipeIndex] -> getSize().y + sizeOfGap / 2 - (birds[i] -> ent.getPosition().y + birds[i] -> ent.getSize().y / 2);
			// horizontal distance to gap
			input(1, 0) = pipes[nextPipeIndex] -> getPosition().x + pipes[nextPipeIndex] -> getSize().x - birds[i] -> ent.getPosition().x;

			result = birds[i] -> n.query(input);
			if(result(0,0) >= 0.5)birds[i] -> velocityY = -15;

		}

		// test if hit the pipes
		for(int i = 0; i < birds.size(); i++){
			for(int k = 0; k < pipes.size(); k++){
				if(pipes[k] -> getGlobalBounds().intersects(birds[i] -> ent.getGlobalBounds()) or
				birds[i] -> ent.getPosition().y >= height or birds[i] -> ent.getPosition().y + birds[i] -> ent.getSize().y <= 0){
					if(birds[i] -> dead)continue;
					// restart
					birds[i] -> dead = true;
					birds[i] -> fitness = distance -
					(pipes[nextPipeIndex] -> getPosition().x + pipes[nextPipeIndex] -> getSize().x - birds[i] -> ent.getPosition().x);
				}
			}
		}

		// if passed a pipe, increment score
		for(int i = 0; i < birds.size(); i++){
			if(birds[i] -> ent.getPosition().x >= pipes[nextPipeIndex] -> getPosition().x + pipes[nextPipeIndex] -> getSize().x
			and pipes.size() != 0 and !birds[i] -> dead){
				score++;
				if(hiscore < score)hiscore = score;
				nextPipeIndex += 2;
				break;
			}

		}


		// if score is divisble by 10, increase speed
		if(score % 10 == 0 and score != 0 and score <= 50 and landmark != score){
			velocityX -= 2;
			landmark = score;
		}



		//updating the scoreboard

		std::stringstream foo1, foo2, foo3, foo4;
		foo1 << generation;
		foo2 << score;
		foo3 << hiscore;
		foo4 << alive;
		scoreboard.setString("Generation: " + foo1.str() +
			"\n" + "Score: " + foo2.str() +
			"\n" + "High Score: " + foo3.str() +
			"\n" + "Alive: " + foo4.str());
		//scoreboard.setPosition((width - scoreboard.getLocalBounds().width) / 2, 100); // adjust allignment of scorebaord


		// updating the display
		window.clear(sf::Color::Black);

		for(int i = 0; i < pipes.size(); i++){
			pipes[i] -> move(velocityX, 0);
			window.draw(*pipes[i]);
		}

		bool allDead = true;
		for(int i = 0; i < birds.size(); i++){
			if(birds[i] -> dead)continue;
			birds[i] -> ent.move(0, birds[i] -> velocityY);
			window.draw(birds[i] -> ent);
			allDead = false;
		}

		marker.setPosition(pipes[nextPipeIndex] -> getPosition().x + pipes[nextPipeIndex] -> getSize().x,
						   pipes[nextPipeIndex] -> getSize().y + sizeOfGap / 2);

		window.draw(marker);

		if(allDead)gameRestart = true;

		distance += -velocityX;
		alive = 0;

		window.draw(scoreboard);

		window.display();

	}

	return 0;
}
