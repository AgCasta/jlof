public class LOF
    {
        /** The training instances  */
        private List<double[]> trainInstances;
        private int numAttributes, numInstances;

        /** The distances among instances. */
        private double[][] distTable;  //jagged array

        /** Indices of the sorted distance */
        private int[][] distSorted;

        /** The minimum values for training instances */
        private double[] minTrain;

        /** The maximum values training instances */
        private double[] maxTrain;
        
        /**
         * @param trainList
         */
        public LOF(List<double[]> trainList)
        {
            // get training data dimensions
            numInstances = trainList.Count();

            double[] first = trainList[0];
            numAttributes = first.Length;

            trainInstances = trainList;

            // get the bounds for numeric attributes of training instances:
            minTrain = new double[numAttributes];
            maxTrain = new double[numAttributes];

            for (int i = 0; i < numAttributes; i++)
            {
                minTrain[i] = Double.PositiveInfinity;
                maxTrain[i] = Double.NegativeInfinity;

                foreach (var instance in trainInstances)
                {
                    if (instance[i] < minTrain[i])
                        minTrain[i] = instance[i];

                    if (instance[i] > maxTrain[i])
                        maxTrain[i] = instance[i];
                }
            }
            
            // fill the table with distances among training instances
            distTable = new double[numInstances + 1][];
            distSorted = new int[numInstances + 1][];
            for (int i = 0; i < numInstances + 1; i++)
            {
                distTable[i] = new double[numInstances + 1];
                distSorted[i] = new int[numInstances + 1];
            }

            int h = 0, j = 0;
            foreach (var instance1 in trainInstances)
            {
                j = 0;
                foreach (var instance2 in trainInstances)
                {
                    distTable[h][j] = getDistance(instance1, instance2);
                    j++;
                }
                if (h == j)
                    distTable[h][j] = -1;
                h++;
            }
        }
        
        /**
         * Returns neighbors for the new example.
         * @param testInstance
         * @param kNN
         * @return
         */
        public List<double[]> getNeighbors(double[] testInstance, int kNN)
        {
            calcuateDistanceToTest(testInstance);

            // get the number of nearest neighbors for the current test instance:
            int numNN = getNNCount(kNN, numInstances);

            int[] nnIndex = new int[numNN];
            for (int i = 1; i <= numNN; i++)
            {
                nnIndex[i - 1] = distSorted[numInstances][i];
            }

            // loop over training data
            List<double[]> res = new List<double[]>(numNN);
            int idx = 0;
            foreach (var instance in trainInstances)
            {
                // check if instance is among neighbors
                for (int j = 0; j < nnIndex.Length; j++)
                {
                    if (nnIndex[j] == idx)
                    {
                        res.Add(instance);
                        break;
                    }
                }
                idx++;
            }
            return res;
        }

        /**
         * Returns LOF score for new example.
         * @param testInstance
         * @param kNN
         * @return
         */
        public double getScore(double[] testInstance, int kNN)
        {
            calcuateDistanceToTest(testInstance);
            return getLofIdx(numInstances, kNN);
        }

        /**
         * Returns LOF scores for training examples.
         * @param kNN
         * @return
         */
        public double[] getTrainingScores(int kNN)
        {
            // update the table with distances among training instances and a fake test instance
            for (int i = 0; i < numInstances; i++)
            {
                distTable[i][numInstances] = Double.MaxValue;
                distSorted[i] = sortedIndices(distTable[i]);
                distTable[numInstances][i] = Double.MaxValue;
            }

            double[] res = new double[numInstances];
            for (int idx = 0; idx < numInstances; idx++)
            {
                res[idx] = getLofIdx(idx, kNN);
            }
            return res;
        }

        private double getLofIdx(int index, int kNN)
        {
            // get the number of nearest neighbors for the current test instance:
            int numNN = getNNCount(kNN, index);

            // get LOF for the current test instance:
            double lof = 0.0;
            for (int i = 1; i <= numNN; i++)
            {
                double lrdi = getLocalReachDensity(kNN, index);
                lof += (lrdi == 0) ? 0 : getLocalReachDensity(kNN, distSorted[index][i]) / lrdi;
            }
            lof /= numNN;
            return lof;
        }

        private void calcuateDistanceToTest(double[] testInstance)
        {
            // update the table with distances among training instances and the current test instance:
            int i = 0;
            foreach (var trainInstance in trainInstances)
            {
                distTable[i][numInstances] = getDistance(trainInstance, testInstance);
                distTable[numInstances][i] = distTable[i][numInstances];
                i++;
            }
            distTable[numInstances][numInstances] = -1;

            // sort the distances
            for (i = 0; i < numInstances + 1; i++)
            {
                distSorted[i] = sortedIndices(distTable[i]);
            }
        }

        private double getDistance(double[] first, double[] second)
        {
            // calculate absolute relative distance
            double distance = 0;
            for (int i = 0; i < this.numAttributes; i++)
            {
                distance += Math.Pow(first[i] - second[i], 2);
            }
            distance = Math.Sqrt(distance);            
            return distance;
        }

        private double getReachDistance(int kNN, int firstIndex, int secondIndex)
        {
            // max({distance to k-th nn of second}, distance(first, second))

            double reachDist = distTable[firstIndex][secondIndex];
            int numNN = getNNCount(kNN, secondIndex);
            if (distTable[secondIndex][distSorted[secondIndex][numNN]] > reachDist)
            {
                reachDist = distTable[secondIndex][distSorted[secondIndex][numNN]];
            }
            return reachDist;
        }

        private int getNNCount(int kNN, int instIndex)
        {
            int numNN = kNN;

            // if there are more neighbors with the same distance, take them too
            for (int i = kNN; i < distTable.Length - 1; i++)
            {
                if (distTable[instIndex][distSorted[instIndex][i]] == distTable[instIndex][distSorted[instIndex][i + 1]])
                    numNN++;
                else
                    break;
            }

            return numNN;
        }

        private double getLocalReachDensity(int kNN, int instIndex)
        {
            // get the number of nearest neighbors:
            int numNN = getNNCount(kNN, instIndex);

            double lrd = 0;

            for (int i = 1; i <= numNN; i++)
            {
                lrd += getReachDistance(kNN, instIndex, distSorted[instIndex][i]);
            }
            lrd = (lrd == 0) ? 0 : numNN / lrd;

            return lrd;
        }

        private int[] sortedIndices(double[] array)
        {
            int[] indices = new int[array.Length];
            for (int i = 0; i < array.Length; i++)
            {
                indices[i] = i;
            }
            Array.Sort(array, indices);
            return indices;
        }

    }
}
