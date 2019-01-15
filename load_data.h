#ifndef LOAD_DATA_H
#define LOAD_DATA_H

#include "feed_data.h"

float* load_data(char* datafile, FeedData* data, void(*mallocFunc)(float**, size_t));
void freeFeedData(FeedData* data);

#endif 
