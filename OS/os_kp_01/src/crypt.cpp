#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <iomanip>
#include "crypt.hpp"

#define LIMIT 10000

ll modpow(ll n, ll p, ll mod)
{
    ll res = 1;
    for (;p;p >>= 1)
    {
        if (p & 1) { res = (1LL * res * n) % mod; }
        n = (1LL * n * n) % mod;
    }
    return res;
}

bool rabin_miller(ll n)
{
    bool flag = true;
    for (ll i = 1; i <= 5 && flag; ++i) 
    {
        ll a = (ll)(rand() + 1);
        ll res = modpow(a, n - 1, n);
        flag &= (res == 1);
    }
    return flag;
}

ll gen_p()
{
    ll gen = (ll)(rand() % LIMIT);
    while (!rabin_miller(gen)) { gen = (ll)(rand() % LIMIT); }
    return gen;
}

ll gcd(ll a, ll b)
{
    while (b)
    {
        ll r = a % b;
        a = b;
        b = r;
    }
    return a;
}

ll gen_c(ll n)
{
    ll gen = (ll)(rand() % LIMIT);
    while (gcd(n, gen) != 1) { gen = (ll)(rand() % LIMIT); }
    return gen;
}

std::pair<ll,ll> euclid(ll a, ll b) 
{
    if(!b) { return std::pair<ll,ll>(1, 0); }
    std::pair<ll,ll> result = euclid(b, a % b);
    return std::pair<ll,ll>(result.second, result.first - (a / b) * result.second);
}

ll mod_inv(ll n, ll mod)
{
    ll inv = euclid(n, mod).first;
    while(inv < 0) { inv += mod; }
    return inv;
}

keys generate_keys()
{
    keys res;
    ll p = gen_p();
    ll q = gen_p();

    ll n = p * q;
    ll f = (p - 1) * (q - 1);
    ll e = gen_c(f);

    res.publ = std::make_pair(n, e);
    ll d = mod_inv(e, f);
    res.priv = std::make_pair(n, d);

    return res;
}

ll encrypt(public_key key, ll val)
    { return modpow(val, key.second, key.first); }

ll decrypt(private_key key, ll val) 
    { return modpow(val, key.second, key.first); }