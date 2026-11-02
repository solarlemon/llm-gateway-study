#define main upstream_cli_main
#include "run.c"
#undef main
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer,argv[1],512);
    int* ids = calloc(strlen(argv[2])+3,sizeof(int));
    int count = 0;
    encode(&tokenizer,argv[2],1,0,ids,&count);
    printf("[");
    for (int i=0; i<count; ++i) printf("%s%d",i ? "," : "",ids[i]);
    puts("]");
    free(ids); free_tokenizer(&tokenizer);
    return 0;
}
