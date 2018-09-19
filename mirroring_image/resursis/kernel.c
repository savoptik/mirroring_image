__kernel void square(__global unsigned char* input, int h, int v)
{
  int i = get_global_id(0);
    int currentRow = i/(v/2);
    int currentCol = i%(v/2);
    if (currentRow >= h) return;
    unsigned char timer = input[(currentRow*v+currentCol)*3];
    unsigned char timeg = input[(currentRow*v+currentCol)*3+1];
    unsigned char timeb = input[(currentRow*v+currentCol)*3+2];
    input[(currentRow*v+currentCol)*3] = input[(currentRow+(v-1)-currentCol)*3];
    input[(currentRow*v+currentCol)*3+1] = input[(currentRow+(v-1)-currentCol)*3+1];
    input[(currentRow*v+currentCol)*3+2] = input[(currentRow+(v-1)-currentCol)*3+2];
    input[(currentRow+(v-1)-currentCol)*3] = timer;
    input[(currentRow+(v-1)-currentCol)*3+1] = timeg;
    input[(currentRow+(v-1)-currentCol)*3+2] = timeb;
}
