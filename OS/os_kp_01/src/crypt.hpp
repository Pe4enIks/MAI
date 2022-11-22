#pragma once

#include <iostream>

typedef long long ll;
typedef std::pair<ll, ll> public_key;
typedef std::pair<ll, ll> private_key;

struct keys
{
    public_key publ;
    private_key priv;
};

keys generate_keys();
ll encrypt(public_key, ll);
ll decrypt(private_key, ll); 

