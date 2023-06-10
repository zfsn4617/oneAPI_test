#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <omp.h>
using namespace std;
using namespace std::chrono;
typedef int64_t int64;
typedef unsigned int uint;

const int64 m1 = 12289;
const int64 m2 = 40961;
const int64 m3 = 65537;
const int64 m4 = 114689;
const int64 M1 = (int64)m2 * m3 * m4;
const int64 M2 = (int64)m1 * m3 * m4;
const int64 M3 = (int64)m1 * m2 * m4;
const int64 M4 = (int64)m1 * m2 * m3;
const int64 m1_ = 11711;
const int64 m2_ = 37278;
const int64 m3_ = 20721;
const int64 m4_ = 94134;
const int64 mod = m1 * m2 * m3 * m4;
const int64 G = 23;
const int64 Gi_1 = 6946;
const int64 Gi_2 = 21371;
const int64 Gi_3 = 45591;
const int64 Gi_4 = 9973;

const int64 limit_inv_1 = 12286;
const int64 limit_inv_2 = 40951;
const int64 limit_inv_3 = 65521;
const int64 limit_inv_4 = 114661;

const int TEST_TIMES = 4000;

const int64 deg = 2048;
int64 limit[TEST_TIMES]; //
int64 L[TEST_TIMES];     //      Ƶ λ  
int64 RR[TEST_TIMES][2 * deg];
int64 a[TEST_TIMES][deg], b[TEST_TIMES][deg];
int64 a_in[TEST_TIMES][2 * deg], b_in[TEST_TIMES][2 * deg];
int64 ret_1[TEST_TIMES][deg], ret_2[TEST_TIMES][deg], ret_3[TEST_TIMES][deg], ret_4[TEST_TIMES][deg];
int64 wn_rec[4][4097];

int64 modpow(int64 a, int64 b, int64 mod)
{
    int64 res = 1;
    while (b)
    {
        if (b & 1)
            res = res * a % mod;
        a = a * a % mod;
        b >>= 1;
    }
    return res % mod;
}

int64 inv(int64 x, int64 mod)
{
    return modpow(x, mod - 2, mod);
}

void NTT(int64* A, int64 type, int64 G, int64 mod, int index)
{
    for (int i = 0; i < limit[index]; ++i)
        if (i < RR[index][i])
            swap(A[i], A[RR[index][i]]);
    int64* wn_rec_1 = NULL;
    switch (mod)
    {
    case m1:
        wn_rec_1 = wn_rec[0];
        break;
    case m2:
        wn_rec_1 = wn_rec[1];
        break;
    case m3:
        wn_rec_1 = wn_rec[2];
        break;
    case m4:
        wn_rec_1 = wn_rec[3];
        break;
    default:
        break;
    }
    for (int mid = 1; mid < limit[index]; mid <<= 1)
    { // ԭ     浥λ  
        int64 wn = modpow(G, (mod - 1) / (mid * 2), mod);
        int64 step = 2048 / mid;
        if (type == -1)
            wn = modpow(wn, mod - 2, mod);

        for (int len = mid << 1, pos = 0; pos < limit[index]; pos += len)
        {
            int64 w = 1;
            for (int k = 0; k < mid; k++, w = (w * wn) % mod)
            {
                if (type == 1)
                    w = wn_rec_1[step * k];
                else
                {
                    w = wn_rec_1[step * (2 * mid - k)];
                    int dg = 0;
                }
                int64 x = A[pos + k];
                int64 y = w * A[pos + mid + k] % mod;
                A[pos + k] = (x + y);         // % mod;
                A[pos + k + mid] = ((x - y)); // +mod) % mod;// (x - y + mod) % mod;
            }
        }
    }
    if (type == -1)
    {
        int64 limit_inv = 0;
        if (mod == m1)
            limit_inv = limit_inv_1;
        if (mod == m2)
            limit_inv = limit_inv_2;
        if (mod == m3)
            limit_inv = limit_inv_3;
        if (mod == m4)
            limit_inv = limit_inv_4;
        for (int i = 0; i < limit[index]; ++i)
            A[i] = (A[i] * limit_inv); //% mod;
    }
}
void poly_mul(int64* ret, int64* a, int64* b, int64 deg1, int64 G, int64 mod, int index)
{
    memset(a_in[index], 0, sizeof(a_in[index]));
    memset(b_in[index], 0, sizeof(b_in[index]));
    memcpy(a_in[index], a, deg * sizeof(int64));
    memcpy(b_in[index], b, deg * sizeof(int64));
    for (limit[index] = 1, L[index] = 0; limit[index] <= deg1; limit[index] <<= 1)
        L[index]++;
    for (int i = 0; i < limit[index]; ++i)
    {
        RR[index][i] = (RR[index][i >> 1] >> 1) | ((i & 1) << (L[index] - 1));
    }
    //   ʼ        

    NTT(a_in[index], 1, G, mod, index);
    NTT(b_in[index], 1, G, mod, index);
    for (int i = 0; i < limit[index]; ++i)
        a_in[index][i] = a_in[index][i] * b_in[index][i] % mod;
    NTT(a_in[index], -1, G, mod, index);

    for (int i = 0; i < deg; i++)
    {
        ret[i] = (a_in[index][i] - a_in[index][deg + i] + mod * 10000000) % mod;
    }
}

int main()
{
    for (int j = 0; j < TEST_TIMES; j++)
    {
        for (int i = 0; i < deg; ++i)
        {
            a[j][i] = 1;
            b[j][i] = 1;
        }
    }
    for (int i = 0; i < 4097; i++)
    {
        wn_rec[0][i] = modpow(G, (m1 - 1) * i / 4096, m1);
        wn_rec[1][i] = modpow(G, (m2 - 1) * i / 4096, m2);
        wn_rec[2][i] = modpow(G, (m3 - 1) * i / 4096, m3);
        wn_rec[3][i] = modpow(G, (m4 - 1) * i / 4096, m4);
    }
    auto start = high_resolution_clock::now();

    for (int i = 0; i < TEST_TIMES; i++)
    {
        poly_mul(ret_1[i], a[i], b[i], 2 * deg - 2, G, m1, i);
        poly_mul(ret_2[i], a[i], b[i], 2 * deg - 2, G, m2, i);
        poly_mul(ret_3[i], a[i], b[i], 2 * deg - 2, G, m3, i);
        poly_mul(ret_4[i], a[i], b[i], 2 * deg - 2, G, m4, i);
    }
    auto end = high_resolution_clock::now();
    cout << "Used Time: " << duration_cast<milliseconds>(end - start).count() << "ms" << endl;
    for (int i = 0; i <= 10; ++i)
    {
        cout << setw(6) << m1 - ret_1[rand() % TEST_TIMES][i];
    }
    cout << endl;
    for (int i = 0; i <= 10; ++i)
    {
        cout << setw(6) << m2 - ret_2[rand() % TEST_TIMES][i];
    }
    cout << endl;
    for (int i = 0; i <= 10; ++i)
    {
        cout << setw(6) << m3 - ret_3[rand() % TEST_TIMES][i];
    }
    cout << endl;
    for (int i = 0; i <= 10; ++i)
    {
        cout << setw(6) << m4 - ret_4[rand() % TEST_TIMES][i];
    }
    cout << endl;
}