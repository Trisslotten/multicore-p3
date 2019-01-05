#pragma once

#include <chrono>
#include <vector>

class Timer
{
	bool paused = false;
	std::chrono::time_point<std::chrono::high_resolution_clock> start;
	std::chrono::time_point<std::chrono::high_resolution_clock> pause_time;
public:
	Timer()
	{
		start = std::chrono::high_resolution_clock::now();
	}
	double restart()
	{
		auto now = std::chrono::high_resolution_clock::now();
		double diff = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count();
		start = std::chrono::high_resolution_clock::now();
		paused = false;
		return diff / 1000000000.0;
	}
	double elapsed()
	{
		auto now = std::chrono::high_resolution_clock::now();
		if (paused)
			now = pause_time;
		double diff = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count();
		return diff / 1000000000.0;
	}
	void pause()
	{
		if (!paused)
		{
			pause_time = std::chrono::high_resolution_clock::now();
			paused = true;
		}
	}
	void resume()
	{
		if (paused)
		{
			auto now = std::chrono::high_resolution_clock::now();
			start += now - pause_time;
			paused = false;
		}
	}
};