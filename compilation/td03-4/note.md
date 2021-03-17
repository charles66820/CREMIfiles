# Compil td03

```yacc
//%type <Expr> a

I: if EXPR instr; // reduce
I: IF EXPR instr else instr // shift
```

## td4 part 3

### Rules

1) E’ → E# // c'est bon quant fin de fichier
2) E → E + B // addition
3) E → E * B // multiplication
4) E → B // un E peut être un B
5) B → nb // B peut être un nombre
• // progression dans une rule

# ?
S(E’) ⊇ { # }

S(E) ⊇ { +, *, # }

S(B) ⊇ { +, *, # }

### Legend

- Lexer token : * + nb #
- End state symboles : E B
- S : shift // Automata transition
- $R_(rules)$ : reduce // Use rules
- G : Go to // Automata transition

| Stats |      *       |      +       |   nb    |      #       |    E    |    B    |
|-------|--------------|--------------|---------|--------------|---------|---------|
| $I_0$ |      --      |      --      | S $I_3$ |      --      | G $I_1$ | G $I_2$ |
| $I_1$ |   S $I_5$    |   S $I_4$    |   --    |      ok      |   --    |   --    |
| $I_2$ | $R_{(4)}=>E$ | $R_{(4)}=>E$ |   --    | $R_{(4)}=>E$ |   --    |   --    |
| $I_3$ | $R_{(5)}=>B$ | $R_{(5)}=>B$ |   --    | $R_{(5)}=>B$ |   --    |   --    |
| $I_4$ |      --      |      --      | S $I_3$ |      --      |   --    | G $I_6$ |
| $I_5$ |      --      |      --      | S $I_3$ |      --      |   --    | G $I_7$ |
| $I_6$ | $R_{(2)}=>E$ | $R_{(2)}=>E$ |   --    | $R_{(2)}=>E$ |   --    |   --    |
| $I_7$ | $R_{(3)}=>E$ | $R_{(3)}=>E$ |   --    | $R_{(3)}=>E$ |   --    |   --    |
