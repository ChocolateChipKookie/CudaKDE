#pragma once
#include <chrono>
#include <map>
#include <string>
#include <thread>
#include <fstream>
#include <algorithm>

namespace kki
{	
	class timer
	{
	public:
		class stopwatch
		{
			std::chrono::time_point<std::chrono::steady_clock> start_time_;
			std::chrono::duration<double> total_{ 0 };

		public:
			void start()
			{
				start_time_ = std::chrono::steady_clock::now();
			}

			void end()
			{

				total_ += std::chrono::steady_clock::now() - start_time_;
			}

			void reset()
			{
				total_ = std::chrono::duration<double>{ 0 };
			}

			std::chrono::duration<double> get_total()
			{
				return total_;
			}

			std::chrono::duration<double> operator ()() const
			{
				return total_;
			}

			explicit operator std::string() const
			{
				return std::to_string(total_.count());
			}

			explicit operator double() const
			{
				return total_.count();
			}
		};

	private:
		std::map<std::string, stopwatch> stopwatches_;

	public:

		stopwatch& operator[](std::string& str)
		{
			return stopwatches_[str];
		}

		stopwatch& operator[](const char* str)
		{
			return stopwatches_[str];
		}

		void reset_all()
		{
			for (auto& sw : stopwatches_)
			{
				sw.second.reset();
			}
		}

		operator std::string()
		{
			std::string total;
			for (auto& sw : stopwatches_)
			{
				total += sw.first + ": " + std::to_string(sw.second().count()) + '\n';
			}
			return total;
		}

		std::string to_string()
		{
			std::string total;
			for (auto& sw : stopwatches_)
			{
				total += sw.first + ": " + std::to_string(sw.second().count()) + '\n';
			}
			return total;
		}

		static timer& global()
		{
			static timer t;
			return t;
		}
	};
}

namespace kki{

	// Basic instrumentation profiler by Cherno

	struct profile_result
	{
		std::string name;
		long long start, end;
		uint32_t thread_ID;
	};

	struct instrumentation_session
	{
		std::string name;
	};

	class instrumentor
	{
	private:
		instrumentation_session* current_session_;
		std::ofstream output_stream_;
		int profile_count_;
	public:
		instrumentor()
			: current_session_(nullptr), profile_count_(0)
		{
		}

		void begin_session(const std::string& name, const std::string& filepath = "results.json")
		{
			output_stream_.open(filepath);
			write_header();
			current_session_ = new instrumentation_session{ name };
		}

		void end_session()
		{
			write_footer();
			output_stream_.close();
			delete current_session_;
			current_session_ = nullptr;
			profile_count_ = 0;
		}

		void write_profile(const profile_result& result)
		{
			if (profile_count_++ > 0)
				output_stream_ << ",";

			std::string name = result.name;
			std::replace(name.begin(), name.end(), '"', '\'');

			output_stream_ << "{";
			output_stream_ << "\"cat\":\"function\",";
			output_stream_ << "\"dur\":" << (result.end - result.start) << ',';
			output_stream_ << "\"name\":\"" << name << "\",";
			output_stream_ << "\"ph\":\"X\",";
			output_stream_ << "\"pid\":0,";
			output_stream_ << "\"tid\":" << result.thread_ID << ",";
			output_stream_ << "\"ts\":" << result.start;
			output_stream_ << "}";

			output_stream_.flush();
		}

		void write_header()
		{
			output_stream_ << "{\"otherData\": {},\"traceEvents\":[";
			output_stream_.flush();
		}

		void write_footer()
		{
			output_stream_ << "]}";
			output_stream_.flush();
		}

		static instrumentor& get()
		{
			static instrumentor instance;
			return instance;
		}
	};

	class instrumentation_timer
	{
	public:
		instrumentation_timer(const char* name)
			: name_(name), stopped_(false)
		{
			auto& i = instrumentor::get();
			start_timepoint_ = std::chrono::high_resolution_clock::now();
		}

		~instrumentation_timer()
		{
			if (!stopped_)
				stop();
		}

		void stop()
		{
			const auto end_timepoint = std::chrono::high_resolution_clock::now();

			const long long start = std::chrono::time_point_cast<std::chrono::microseconds>(start_timepoint_).time_since_epoch().count();
			const long long end = std::chrono::time_point_cast<std::chrono::microseconds>(end_timepoint).time_since_epoch().count();

			const uint32_t threadID = std::hash<std::thread::id>{}(std::this_thread::get_id());
			instrumentor::get().write_profile({ name_, start, end, threadID });

			stopped_ = true;
		}
	private:
		const char* name_;
		std::chrono::time_point<std::chrono::high_resolution_clock> start_timepoint_;
		bool stopped_;
	};
}

#ifdef KKI_PROFILE
	#define KKI_PROFILE_BEGIN(name) kki::instrumentor::get().begin_session(name)
	#define KKI_PROFILE_END() kki::instrumentor::get().end_session()
	#define KKI_PROFILE_SCOPE(name) kki::instrumentation_timer timer##__LINE__(name)
	#define KKI_PROFILE_FUNCTION() KKI_PROFILE_SCOPE(__FUNCSIG__)
#else
	#define KKI_PROFILE_BEGIN(name)
	#define KKI_PROFILE_END()
	#define KKI_PROFILE_SCOPE(name) 
	#define KKI_PROFILE_FUNCTION()
#endif
