#include<time.h>
#include<stdlib.h>
#include<stdio.h>

int main() {
    srand(time(NULL));
    float init = 25088.574773;
    char filename[10] = "cputime";
    FILE *fp = fopen(filename, "a");

    for (int i = 0; i < 50; i++) {
        float w = ((float) rand() / RAND_MAX / 500.0) - 0.001 + 1.0;
        printf("%f ", w);
        fprintf(fp, "%2.6f ", init * w);
    }

    fclose(fp);
    

    return 0;
}