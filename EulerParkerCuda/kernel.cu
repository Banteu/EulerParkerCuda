
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <device_atomic_functions.h>
#include <algorithm>
#include <vector>
#include <Windows.h>
#include <cstring>
using std::vector;
using namespace std;
#define setBit(a, n) (a = a | (1 << n))
#define offBit(a, n) (a = a & (~(1 << n)))
#define getBit(a, n) (a & (1 << n))

#define setOccupation(a, b)\
a.x |= b.x;\
a.y |= b.y;\
a.z |= b.z;\
a.w |= b.w\

#define releaseOccupation(a, b)\
a.x  = a.x & (~b.x);\
a.y  = a.y & (~b.y);\
a.z  = a.z & (~b.z);\
a.w  = a.w & (~b.w)\

#define DIM 10
#define getIndex(i, j) (DIM * (i) + j)

#define FIND_DIAGONALS true

void printCPUName(FILE* out)
{
	int info[4] = { -1 };
	__cpuid(info, 0x80000000);
	unsigned int nExIds = info[0];

	char retString[0x40] = { 0 };
	for (unsigned int i = 0x80000000; i <= nExIds; ++i)
	{
		__cpuid(info, i);
		if (i == 0x80000002)
		{
			memcpy(retString, info, sizeof(info));
		}
		else if (i == 0x80000003)
		{
			memcpy(retString + 16, info, sizeof(info));
		}
		else if (i == 0x80000004)
		{
			memcpy(retString + 32, info, sizeof(info));
		}
	}
	fprintf(out, "%s \n \n \n \n", retString);
}

bool setDeviceAndPrintPcConfigurations(FILE* out)
{
	int cudaDevicesCount = 0;
	cudaGetDeviceCount(&cudaDevicesCount);
	cudaDeviceProp prop;
	int lastMultiprocessors = 0;
	int computeDevice = -1;

	for (int i = 0; i < cudaDevicesCount; ++i)
	{
		cudaGetDeviceProperties(&prop, i);
		if (prop.multiProcessorCount > lastMultiprocessors)
		{
			prop.multiProcessorCount > lastMultiprocessors;
			computeDevice = i;
		}
	}
	if (computeDevice == -1)
	{
		fprintf(out, "There no any compute device... \n");
		return false;
	}

	cudaGetDeviceProperties(&prop, computeDevice);
	fprintf(out, "Compute device: %s\n", prop.name);
	fprintf(out, "Multiprocessors count: %d\n", prop.multiProcessorCount);
	fprintf(out, "Regs per multiprocessor: %d\n", prop.regsPerMultiprocessor);


	fprintf(out, "CPU INFO: \n");
	printCPUName(out);
	cudaSetDevice(computeDevice);
	return true;
}

void oldReadDlsFromFile(string dls_file_name, vector<vector<vector<int>>> &dls_vec)
{
	string str;
	ifstream dls_file(dls_file_name.c_str());
	if (!dls_file.is_open()) {
		cerr << "can't open " << dls_file_name.c_str() << endl;
		exit(1);
	}

	stringstream sstream;
	int cell_val;
	vector<int> dls_row;
	vector<vector<int>> dls;
	while (getline(dls_file, str)) {
		if (str.size() < 10) continue;
		sstream << str;
		while (sstream >> cell_val) {
			dls_row.push_back(cell_val);
			if (dls_row.size() == 10) {
				dls.push_back(dls_row);
				dls_row.clear();
			}
			if (dls.size() == 10) {
				dls_vec.push_back(dls);
				dls.clear();
			}
		}
		sstream.clear();
		sstream.str("");
	}

	dls_file.close();
}

__device__ bool getNextIj(int& i, int& j, int* lineToColumn, int& occupiedColumns, int& occupiedValue, char* matrix)
{
	while (j < DIM)
	{
		++j;
		if (j == DIM)
		{
			--i;
			if (i < 0)
			{
				break;
			}
			else
			{
				offBit(occupiedColumns, lineToColumn[i]);
				j = lineToColumn[i];
				offBit(occupiedValue, matrix[getIndex(i, j)]);

			}
		}
		else
		{
			if (getBit(occupiedColumns, j) == 0 && getBit(occupiedValue, matrix[getIndex(i, j)]) == 0)
			{
				setBit(occupiedValue, matrix[getIndex(i, j)]);
				setBit(occupiedColumns, j);
				break;
			}
		}
	}
	return i >= 0;
}

__device__ bool initTransversal(char* matrix, int* lineToColumn, int& occupiedColumns, int& occupiedValue)
{
	int i = 0;
	int j = -1;

	for (i = 0; i < DIM; ++i)
	{
		lineToColumn[i] = 0;
	}
	i = 0;
	occupiedColumns = 0;
	occupiedValue = 0;
	int firstColumn = threadIdx.x / 10;
	lineToColumn[0] = firstColumn;
	setBit(occupiedColumns, firstColumn);
	setBit(occupiedValue, matrix[getIndex(0, firstColumn)]);
	int secondLineColumn = threadIdx.x % 10;
	if (firstColumn == secondLineColumn || getBit(occupiedValue, matrix[getIndex(1, secondLineColumn)]))
		return false;
	lineToColumn[1] = secondLineColumn;
	setBit(occupiedColumns, secondLineColumn);
	setBit(occupiedValue, matrix[getIndex(1, secondLineColumn)]);
	i = 2;
	j = -1;


	while ((i < 10) && getNextIj(i, j, lineToColumn, occupiedColumns, occupiedValue, matrix))
	{
		lineToColumn[i] = j;
		++i;
		j = -1;
	}
	return i == DIM;
}

__device__ bool getNextTransversal(char* matrix, int* lineToColumn, int& occupiedColumns, int& occupiedValue)
{
	offBit(occupiedColumns, lineToColumn[DIM - 1]);
	offBit(occupiedValue, matrix[getIndex(DIM - 1, lineToColumn[DIM - 1])]);

	int i = DIM - 2;
	int j = lineToColumn[i];

	offBit(occupiedColumns, j);
	int index = getIndex(i, j);
	int offVal = matrix[index];
	offBit(occupiedValue, offVal);

	bool found = false;
	int a = threadIdx.x / 10;
	int b = threadIdx.x % 10;
	while (getNextIj(i, j, lineToColumn, occupiedColumns, occupiedValue, matrix))
	{

		lineToColumn[i] = j;
		if (lineToColumn[0] != a || lineToColumn[1] != b)
			break;
		++i;
		j = -1;
		if (i == DIM)
		{
			if (!FIND_DIAGONALS)
			{
				found = true; break;
			}
			else
			{
				int onMainDiag = 0;
				int onAltDiag = 0;
				for (char p = 0; p < DIM; ++p)
				{
					if (lineToColumn[p] == p) ++onMainDiag;
					if (lineToColumn[p] == DIM - p - 1) ++onAltDiag;
				}
				if (onMainDiag == 1 && onAltDiag == 1)
				{
					found = true;
					break;
				}
			}
		}
	}
	return found;
}

__device__ void findAllTransversalls(char* matrix, char* output, int* totalCount, int squareId, int* resultIndexer)
{

	int lineToColumn[10] = { 0 };
	int occupiedColumns = 0;
	int occupiedValues = 0;
	if (!initTransversal(matrix, lineToColumn, occupiedColumns, occupiedValues))
	{
		return;
	}
	int onMainDiag = 0;
	int onAltDiag = 0;
	if (FIND_DIAGONALS)
	{
		for (int p = 0; p < DIM; ++p)
		{
			if (lineToColumn[p] == p) ++onMainDiag;
			if (lineToColumn[p] == DIM - p - 1) ++onAltDiag;
		}
	}
	if (!FIND_DIAGONALS || (onMainDiag == 1 && onAltDiag == 1))
	{
		int count = atomicAdd(totalCount, 1);
		resultIndexer[count] = squareId;
		for (int i = 0; i < DIM; ++i)
		{
			output[DIM * (count)+i] = lineToColumn[i];
		}

		//++(*totalCount);
	}
	while (getNextTransversal(matrix, lineToColumn, occupiedColumns, occupiedValues))
	{
		int count = atomicAdd(totalCount, 1);
		resultIndexer[count] = squareId;
		for (int i = 0; i < DIM; ++i)
		{
			output[DIM * (count)+i] = lineToColumn[i];
		}
		//++(*totalCount);
	}
}
#define BLOCK_DIM 100
__global__ void transversalSearch(int offset, char* matrices, char* output, int* resultsCounter, int matrixCount, int* resultIndexer)
{
	__shared__ char sharedMatrices[100];

	int squareId = blockIdx.x + offset;
	if (squareId >= matrixCount)
	{
		return;
	}
	sharedMatrices[threadIdx.x] = matrices[squareId * DIM * DIM + threadIdx.x];
	__syncthreads();
	findAllTransversalls(sharedMatrices, output, resultsCounter, squareId, resultIndexer);
}

__global__ void cudaOrtMatrixSearch(int startTransvOffset, int firstTransvRange, int matrixCount, char* transversals, int* transvOffsets, int* matrixIndexer, int* resultsCounter, int4* occupation, int* resultOrts)
{
	__shared__ int ortMatricesShared[128 * 10];

	int* ortMatrix = ortMatricesShared + threadIdx.x * 10;
	//int ortMatrix[10];
	for (int i = 0; i < 10; ++i)
		ortMatrix[i] = -1;
	int firstTransv = threadIdx.x + blockIdx.x * blockDim.x + startTransvOffset;
	if (firstTransv >= firstTransvRange)
	{
		return;
	}
	ortMatrix[0] = firstTransv;
	int4 occupied = occupation[firstTransv];
	int matrix = matrixIndexer[firstTransv];
	int indexToMove = 1;
	for (;;)
	{
		if (ortMatrix[indexToMove] == -1)
		{
			int nextIndex = transvOffsets[matrix + indexToMove * matrixCount];
			if (nextIndex == transvOffsets[matrix + indexToMove * matrixCount + 1])
			{
				break;
			}
			ortMatrix[indexToMove] = nextIndex;
		}
		else
		{
			ortMatrix[indexToMove]++;
			if (ortMatrix[indexToMove] == transvOffsets[matrix + indexToMove * matrixCount + 1])
			{
				if (indexToMove == 1)
				{
					break;
				}
				else
				{
					ortMatrix[indexToMove] = -1;
					--indexToMove;
					releaseOccupation(occupied, occupation[ortMatrix[indexToMove]]);
					continue;
				}
			}
		}
		int4 locOccupation = occupation[ortMatrix[indexToMove]];
		if (((occupied.x & locOccupation.x) | (occupied.y & locOccupation.y) |
			(occupied.z & locOccupation.z) | (occupied.w & locOccupation.w)) == 0)
		{
			++indexToMove;
			setOccupation(occupied, locOccupation);
		}
		if (indexToMove == 10)
		{
			int ind = atomicAdd(resultsCounter, 1);
			int* pntr = resultOrts + ind * 10;
			*pntr++ = ortMatrix[0]; *pntr++ = ortMatrix[1];
			*pntr++ = ortMatrix[2]; *pntr++ = ortMatrix[3];
			*pntr++ = ortMatrix[4]; *pntr++ = ortMatrix[5];
			*pntr++ = ortMatrix[6]; *pntr++ = ortMatrix[7];
			*pntr++ = ortMatrix[8]; *pntr++ = ortMatrix[9];
			indexToMove--;
			releaseOccupation(occupied, occupation[ortMatrix[indexToMove]]);
		}

	}
};



/*
Profiler
*/

class PerfCounter
{
public:
	PerfCounter()
	{
		QueryPerformanceCounter(&lastTime);
		LARGE_INTEGER freq;
		QueryPerformanceFrequency(&freq);
		divisor = 1.0 / double(freq.QuadPart);
	};
	void setStart()
	{
		QueryPerformanceCounter(&lastTime);
	}
	double getElapsedMilliseconds()
	{
		LARGE_INTEGER currTime;
		QueryPerformanceCounter(&currTime);
		double time = double(currTime.QuadPart) - double(lastTime.QuadPart);
		return time * divisor * 1000;
	}
	double getElapsedMillisecondsAndSetStart()
	{
		LARGE_INTEGER currTime;
		QueryPerformanceCounter(&currTime);
		double time = double(currTime.QuadPart) - double(lastTime.QuadPart);
		lastTime = currTime;
		return time * divisor * 1000;
	}
private:
	LARGE_INTEGER lastTime;
	double divisor;
};

using namespace std;

void skipLine(FILE* stream)
{
	char fl = fgetc(stream);
	while (fl != '\n')
	{
		fl = fgetc(stream);
	};
}

int readDlsFromFile(FILE* stream, vector<vector<vector<int>>> &dls_vec, int count)
{
	dls_vec.resize(count, std::vector<std::vector<int > >(10, std::vector<int>(10, 0)));
	int coll = 0;
	bool okRead = true;
	for (int i = 0; i < count && okRead; ++i)
	{
		for (int j = 0; j < 10 && okRead; ++j)
		{
			for (int k = 0; k < 10 && okRead; ++k)
			{
				if (fscanf(stream, "%d ", &dls_vec[i][j][k]) != 1)
				{
					okRead = false;
				}
			}
		}
		if (!okRead)
		{
			break;
		}
		++coll;
	}
	dls_vec.resize(coll);

	return coll;
}


#define MAX_MATRICES_TO_READ 64 * 2048

int main()
{

	FILE* log = fopen("GPUlog.txt", "w");
	//	freopen("log.txt", "w", stdout);
	if (!setDeviceAndPrintPcConfigurations(log))
	{
		return 0;
	}

	/// Allocate CPU data storages
	std::vector<char> serializedMatrices(DIM * DIM * MAX_MATRICES_TO_READ);
	const int resultBufferSize = 200 * 1000 * 1000;	// Why not?
	const int indexerBufferSize = resultBufferSize / 10;
	char*	  output = reinterpret_cast<char*>(malloc(sizeof(char) * resultBufferSize));
	int*	  indexer = reinterpret_cast<int*>(malloc(sizeof(int) * indexerBufferSize));
	int		  totalCount = 0;

	int		  globalResultOffset = 0;

	/// Allocate GPU memory and send all data to GPU ///

	char*	 cudaResults = NULL;
	char*	 cudaMatrices = NULL;
	int*	 cudaResultsCounter = NULL;
	int*	 resultIndexer = NULL;
	int*	 cudaTransversallsOffsets = NULL;
	int* cudaOrts;
	PerfCounter time;
	PerfCounter totalComputeTimer;

	int totalTransversalsCount = 0;
	int totalOrtsCount = 0;
	double overallReadingDataTime = 0;
	/**
	Allocate GPU memory
	*/

	cudaMalloc(&cudaOrts, sizeof(int) * 10000); /// I think this is enough space?
	cudaMalloc(&cudaResults, sizeof(char) * resultBufferSize);
	cudaMalloc(&resultIndexer, sizeof(int) * indexerBufferSize);
	cudaMalloc(&cudaResultsCounter, sizeof(int));
	cudaMalloc(&cudaMatrices, sizeof(char) * MAX_MATRICES_TO_READ * DIM * DIM);
	cudaMalloc(&cudaTransversallsOffsets, sizeof(int) * (MAX_MATRICES_TO_READ * 10 + 1));
	fprintf(log, "GPU mem allocation time: %lf \n", time.getElapsedMillisecondsAndSetStart());


	std::vector<std::vector<std::vector<int> > > input;
	//vector<vector<int>> a{
	//	{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 },
	//	{ 2, 3, 4, 9, 8, 1, 0, 5, 6, 7 },
	//	{ 3, 4, 9, 8, 2, 7, 1, 0, 5, 6 },
	//	{ 8, 7, 6, 5, 0, 9, 4, 3, 2, 1 },
	//	{ 5, 0, 1, 7, 6, 3, 2, 8, 9, 4 },
	//	{ 6, 5, 0, 1, 7, 2, 8, 9, 4, 3 },
	//	{ 4, 9, 8, 2, 3, 6, 7, 1, 0, 5 },
	//	{ 7, 6, 5, 0, 1, 8, 9, 4, 3, 2 },
	//	{ 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 },
	//	{ 1, 2, 3, 4, 9, 0, 5, 6, 7, 8 }
	//};
	FILE* outputStream = fopen("GPUresult.txt", "w");
	FILE* stream = fopen("input.txt", "r");
	skipLine(stream);
	skipLine(stream);
	int totalMatricesCount = 0;
	for (int restart = 0; restart < 200; ++restart)
	{
		printf("Search restart %d - maximum 200...\n", restart + 1);
		PerfCounter internalTime;
		time.setStart();
		if (!readDlsFromFile(stream, input, MAX_MATRICES_TO_READ))
		{
			break;
		};
		overallReadingDataTime += time.getElapsedMilliseconds();
		fprintf(log, "Read new DLS from file: %lf \n", time.getElapsedMillisecondsAndSetStart());
		totalMatricesCount += input.size();
		fprintf(log, "Readed count: %d Total computed count: %d \n", input.size(), totalMatricesCount);
		////oldReadDlsFromFile("Squares.txt", input);
		//for (int i = 0; i < 30; ++i)
		//{
		//	int vl = rand() % (input.size() - 1);
		//	input[vl] = a;
		//}


		/**
		Build serialized matrices vector, needed for sending on GPU
		*/

		int matricesCount = input.size();
		int p = 0;
		for (int m = 0; m < input.size(); ++m)
		{
			for (int i = 0; i < DIM; ++i)
			{
				for (int j = 0; j < DIM; ++j)
				{
					serializedMatrices[p] = input[m][i][j];
					++p;
				}
			}
		}
		/*
		Release input data
		*/
		input.clear();
		/**
		Send data to GPU
		*/
		time.setStart();
		cudaMemcpy(cudaMatrices, serializedMatrices.data(), sizeof(char) * DIM * DIM * matricesCount, cudaMemcpyHostToDevice);
		cudaMemset(cudaResultsCounter, 0, sizeof(int));
		fprintf(log, "Send data to GPU time: %lf \n", time.getElapsedMillisecondsAndSetStart());
		PerfCounter trTime;
		int countPerKernelStart = 1024;
		for (int pass = 0; pass < matricesCount; pass += countPerKernelStart)
		{
			transversalSearch << < countPerKernelStart, BLOCK_DIM >> >(pass, cudaMatrices, cudaResults, cudaResultsCounter, matricesCount, resultIndexer);
			cudaDeviceSynchronize();
			//	cudaMemcpy(&totalCount, cudaResultsCounter, sizeof(int), cudaMemcpyDeviceToHost);
			//	printf("Transversal search, matrices %d - %d, Total count %d, Time %lf \n", pass, pass + countPerKernelStart, totalCount, trTime.getElapsedMillisecondsAndSetStart());

		}
		fprintf(log, "Transversal search time: %lf \n", time.getElapsedMillisecondsAndSetStart());

		/**
		Copying data back from GPU
		*/

		cudaMemcpy(&totalCount, cudaResultsCounter, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(output, cudaResults, sizeof(char) * resultBufferSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(indexer, resultIndexer, sizeof(int) * indexerBufferSize, cudaMemcpyDeviceToHost);
		fprintf(log, "Copy data from GPU to CPU time: %lf \n", time.getElapsedMillisecondsAndSetStart());
		totalTransversalsCount += totalCount;

		/**
		Rearranging transversals in order to have transversals beginning from column = 0 at first, column = 1 next, e.t.c
		*/
		std::vector<int> counts(10, 0);
		std::vector<std::vector<int> > countsPerMatrix(matricesCount, std::vector<int>(10, 0));
		std::vector<std::vector<int> > transversalsOffsets(matricesCount - 1, std::vector<int>(10, 0));
		std::vector<int> serializedOffsets((matricesCount - 1) * 10 + 11);
		transversalsOffsets.push_back(std::vector<int>(11, 0));
		for (int i = 0; i < totalCount; ++i)
		{
			counts[output[i * DIM]]++;
			countsPerMatrix[indexer[i]][output[i * DIM]]++;
		}
		int offset = 0;
		for (int i = 0; i < DIM; ++i)
		{
			for (int m = 0; m < matricesCount; ++m)
			{
				transversalsOffsets[m][i] = offset;
				offset += countsPerMatrix[m][i];
			}
		}
		transversalsOffsets[matricesCount - 1][DIM] = offset;
		for (int j = 0; j < 10; ++j)
		{
			for (int i = 0; i < matricesCount; ++i)
			{
				serializedOffsets[j * matricesCount + i] = transversalsOffsets[i][j];
			}
		}
		serializedOffsets[(matricesCount - 1) * 10 + 10] = transversalsOffsets[matricesCount - 1][10];
		std::vector<std::vector<int> > copied = transversalsOffsets;
		std::vector<char> rearrangedTransversals(totalCount * 10);
		std::vector<int> newIndexer(totalCount);
		std::vector<int4> occupiedByVector;
		occupiedByVector.resize(totalCount);

		/**
		Augment each transversal with bool matrix, which shows cells which this transversal ocuppies
		*/

		for (int i = 0; i < totalCount; ++i)
		{
			int matr = indexer[i];
			int arr = output[i * DIM];
			newIndexer[copied[matr][arr]] = matr;
			memcpy(rearrangedTransversals.data() + copied[matr][arr] * DIM, output + i * DIM, sizeof(char) * DIM);
			int4 occupied = make_int4(0, 0, 0, 0);
			for (int j = 0; j < DIM; ++j)
			{
				int vl = 10 * j + output[i * DIM + j];
				if (vl < 25)
				{
					occupied.x |= (1 << vl);
					continue;
				}
				if (vl < 50)
				{
					vl -= 25;
					occupied.y |= (1 << vl);
					continue;
				}
				if (vl < 75)
				{
					vl -= 50;
					occupied.z |= (1 << vl);
					continue;
				}
				vl -= 75;
				occupied.w |= (1 << vl);
			}
			occupiedByVector[copied[matr][arr]] = occupied;
			copied[matr][arr]++;
		}
		fprintf(log, "Prepare transversals time: %lf \n", time.getElapsedMillisecondsAndSetStart());
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////

		/// Send transversals to GPU ///
		int4* cudaOccupation;
		cudaMalloc(&cudaOccupation, sizeof(int4) * occupiedByVector.size());
		fprintf(log, "New GPU mem allocs time: %lf \n", time.getElapsedMillisecondsAndSetStart());

		cudaMemcpy(cudaResults, rearrangedTransversals.data(), rearrangedTransversals.size() * sizeof(char), cudaMemcpyHostToDevice);
		cudaMemcpy(resultIndexer, newIndexer.data(), newIndexer.size() * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(cudaTransversallsOffsets, serializedOffsets.data(), serializedOffsets.size() * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(cudaOccupation, occupiedByVector.data(), sizeof(int4)  * occupiedByVector.size(), cudaMemcpyHostToDevice);
		cudaMemset(cudaResultsCounter, 0, sizeof(int));
		fprintf(log, "Send data to GPU time: %lf \n", time.getElapsedMillisecondsAndSetStart());


		/*
		Search for orts...
		*/
		fprintf(log, "Transversals at first cell: %d \n", counts[0]);
		for (int ft = 0; ft < counts[0]; ft += 1024 * 128)
		{

			cudaOrtMatrixSearch << <1024, 128 >> >(ft, counts[0], matricesCount, cudaResults, cudaTransversallsOffsets, resultIndexer, cudaResultsCounter, cudaOccupation, cudaOrts);
			cudaDeviceSynchronize();
			//	printf("Ort search %d - %d \n", ft, ft + 1024 * 128);
		}
		fprintf(log, "Ort search time: %lf \n", time.getElapsedMillisecondsAndSetStart());

		/*
		Recieve results...
		*/

		int ortsCount = 0;
		std::vector<int> resultOrts(1000 * 10, 0);
		cudaMemcpy(&ortsCount, cudaResultsCounter, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&resultOrts[0], cudaOrts, sizeof(int) * 10000, cudaMemcpyDeviceToHost);

		fprintf(log, "Ort copy to host time: %lf \n", time.getElapsedMillisecondsAndSetStart());

		std::vector<std::vector<std::vector<int> > > resultOrtsMatrices;
		std::vector<int> resultIds;
		for (int i = 0; i < ortsCount; ++i)
		{
			resultOrtsMatrices.push_back(std::vector<std::vector<int> >(10, std::vector<int>(10, 0)));
			resultIds.push_back(newIndexer[resultOrts[i * 10]] + globalResultOffset);
			for (int k = 0; k < 10; ++k)
			{
				for (int l = 0; l < 10; ++l)
				{
					resultOrtsMatrices[i][l][rearrangedTransversals[10 * resultOrts[i * 10 + k] + l]] = k;
				}
			}
		}
		fprintf(log, "Orts found: %d \n", ortsCount);
		totalOrtsCount += ortsCount;
		globalResultOffset += matricesCount;
		fprintf(log, "One cycle internal time: %lf \n", internalTime.getElapsedMillisecondsAndSetStart());


		for (int i = 0; i < ortsCount; ++i)
		{
			fprintf(outputStream, " ===========Ortogonal to ID: %d================\n", resultIds[i]);
			for (int s = 0; s < 10; ++s)
			{
				for (int j = 0; j < 10; ++j)
				{
					fprintf(outputStream, "%d ", resultOrtsMatrices[i][s][j]);
				}
				fprintf(outputStream, "\n");
			}
			fprintf(outputStream, " ===============================================\n");
		}
		fprintf(log, "Save results time: %lf \n", internalTime.getElapsedMillisecondsAndSetStart());
		cudaFree(cudaOccupation);
	}
	/// Release memory ///
	cudaFree(cudaResults);
	cudaFree(resultIndexer);
	cudaFree(cudaResultsCounter);
	cudaFree(cudaMatrices);


	fclose(outputStream);
	fclose(stream);

	fprintf(log, "Found transversals total count: %d \n", totalTransversalsCount);
	fprintf(log, "Found orts total count: %d \n", totalOrtsCount);
	fprintf(log, "Reading data time: %lf \n", overallReadingDataTime);
	fprintf(log, "Total compute timer: %lf \n", totalComputeTimer.getElapsedMillisecondsAndSetStart() - overallReadingDataTime);
	fclose(log);
	printf("Done\n");
	return 0;
}