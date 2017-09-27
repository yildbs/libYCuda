#include <chrono>
#include <map>
#include <vector>

namespace ycuda{

class YTimeCounter{
private:
	std::map<std::string, std::chrono::high_resolution_clock::time_point> start_times;
	std::map<std::string, double> elapsed_times;

public:
	void Start(std::string name){
		this->start_times[name] = std::chrono::high_resolution_clock::now();
	}
	void End(std::string name){
		std::chrono::high_resolution_clock::time_point end= std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed= end - this->start_times[name];
		this->elapsed_times[name] = elapsed.count() / 1000.;
	}
	void DeleteAllTimes(){
		this->start_times.clear();
		this->elapsed_times.clear();
	}
	std::vector<std::string> GetNames(std::vector<std::string> buffer){
		buffer.clear();
		for (auto key : this->elapsed_times){
			buffer.push_back(key.first);
		}
		return buffer;
	}
	double GetElapsedTime(std::string name){
		double elapsed = 0;
		try{
			elapsed = this->elapsed_times[name];
		}
		catch (...){
			printf("Error occured %s, %d\n", __FILE__, __LINE__);
		}
		return elapsed;
	}

};

}