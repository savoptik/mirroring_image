__kernel void square(__global unsigned char* input, int h, int v)
{
  int i = get_global_id(0);
    int currentRow = i/(v/2);
    int currentCol = i%(v/2);
    if (currentRow >= h) return;
    int indFirst = (currentRow * v + currentCol) * 3;
    int indLast = (currentRow*v+(v-1)-currentCol)*3;
    unsigned char timer = input[indFirst];
    unsigned char timeg = input[indFirst+1];
    unsigned char timeb = input[indFirst+2];
    input[indFirst] = input[indLast];
    input[indFirst+1] = input[indLast+1];
    input[indFirst+2] = input[indLast+2];
    input[indLast] = timer;
    input[indLast+1] = timeg;
    input[indLast+2] = timeb;
}
