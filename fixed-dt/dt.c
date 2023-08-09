/**
 * @author      : t-hara (t-hara@$HOSTNAME)
 * @file        : dt
 * @created     : Saturday Nov 12, 2022 04:33:59 JST
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#define MAX_TREE_DEPTH 20
#define TREE_LEAF -1
#define TREE_UNDEFINED -2
#define FIXED_POINT_DIGITS 16
#define NUM_FEATURES 12
#define abs(x) ((x)<0 ? -(x) : (x))

void dt(int64_t *feat, int64_t *childrenLeft, int64_t *childrenRight, int64_t *value, int64_t *feature, int64_t *threshold, unsigned int *class_indices) {
  int current_node = 0;
  int i;
  for (i = 0; i < MAX_TREE_DEPTH; i++) {
    int64_t current_left_child  = childrenLeft[current_node];
    int64_t current_right_child = childrenRight[current_node];
    int64_t current_feature     = feature[current_node];
    int64_t current_threshold   = threshold[current_node];
    if (current_left_child == TREE_LEAF || current_feature == TREE_UNDEFINED) {
      break;
    } else {
      if (current_feature >= 0 && current_feature < NUM_FEATURES ) {
        int64_t current_feature_value = feat[current_feature];
        if (current_feature_value <= current_threshold) {
          current_node = (int) current_left_child;
        } else {
          current_node = (int) current_right_child;
        }
      }
    }
  }
  int64_t current_value = value[current_node];
  /* printf("%lld\t%d\n", current_value, current_node); */
  class_indices[0] = current_value;
  /* return current_value; */
}

void random_forest(int64_t *feat, int64_t *childrenLeft, int64_t *childrenRight, int64_t *value, int64_t *feature, int64_t *threshold, unsigned int *class_indices, unsigned int num_classes, unsigned int NUM_ESTIMATORS, unsigned int N) {
  int current_node, i, j, accumulator[num_classes];
  int argmax = 0;
  int max_val = 0;
  int64_t current_left_child, current_right_child, current_feature, current_threshold, current_value, current_feature_value;
  for (j = 0; j < num_classes; j++) {
    accumulator[j] = 0;
  }
  for (j = 0; j < NUM_ESTIMATORS; j++) {
    current_node = 0;
    for (i = 0; i < MAX_TREE_DEPTH; i++) {
      current_left_child  = childrenLeft[j*N + current_node];
      current_right_child = childrenRight[j*N + current_node];
      current_feature     = feature[j*N + current_node];
      current_threshold   = threshold[j*N + current_node];
      if (current_right_child == TREE_LEAF || current_left_child == TREE_LEAF || current_threshold == TREE_UNDEFINED || current_feature == TREE_UNDEFINED) {
        break;
      } else {
        if (current_feature >= 0 && current_feature < NUM_FEATURES ) {
          current_feature_value = feat[current_feature];
          if (current_feature_value <= current_threshold) {
            current_node = (int) current_left_child;
          } else {
            current_node = (int) current_right_child;
          }
        }
      }
    }
    current_value = value[j*N + current_node];
    accumulator[current_value]++;
  }
  for (j = 0; j < num_classes; j++) {
    if (accumulator[j] > max_val) {
      argmax = j;
      max_val = accumulator[j];
    }
  }
  /* printf("argmax %d\n", argmax); */
  /* printf("%lld\t%d\n", current_value, current_node); */
  class_indices[0] = argmax;
  /* return current_value; */
}
