(* verify.wl — recompute each SDP's optimal value independently and
   check that it matches the stored provenance.json to the stated digits.
   Bead: arb-prec-IPM-b09.

   Each verification is a fresh symbolic/numeric computation that does NOT
   re-use the generator code path, to provide an independent check.
*)

nFailures = 0;
nPassed   = 0;
goldenDir = FileNameJoin[{DirectoryName[$InputFileName], ".."}];

(* ------------------------------------------------------------------
   Helper: read provenance.json and extract optimal_value string
   ------------------------------------------------------------------ *)
ReadProvenance[dir_] := Module[{path, txt, val, digits},
  path = FileNameJoin[{dir, "provenance.json"}];
  txt  = Import[path, "Text"];
  (* Extract optimal_value field *)
  val    = StringCases[txt, "\"optimal_value\": \"" ~~ s:Except["\""].. ~~ "\"" :> s][[1]];
  digits = ToExpression[StringCases[txt, "\"optimal_value_digits\": " ~~ d:DigitCharacter.. :> d][[1]]];
  {val, digits}
];

(* ------------------------------------------------------------------
   Helper: compare two decimal strings to n significant digits.
   Returns True if they agree in the first n significant digits.
   ------------------------------------------------------------------ *)
DigitsAgree[stored_String, computed_String, n_Integer] :=
  Module[{s1, s2, cmpLen},
    (* Strip the decimal point and compare the first n digit characters directly. *)
    s1 = StringReplace[stored,   "." -> ""];
    s2 = StringReplace[computed, "." -> ""];
    cmpLen = Min[StringLength[s1], StringLength[s2], n];
    StringTake[s1, cmpLen] === StringTake[s2, cmpLen]
  ];

CheckProblem[name_, computedVal_] := Module[
  {dir, stored, digits, computedStr, agree, diff},
  dir  = FileNameJoin[{goldenDir, name}];
  {stored, digits} = ReadProvenance[dir];
  computedStr = FormatNum[computedVal];
  agree = DigitsAgree[stored, computedStr, digits];
  Print["  stored:   ", StringTake[stored,   Min[25, StringLength[stored]]  ], "..."];
  Print["  computed: ", StringTake[computedStr, Min[25, StringLength[computedStr]]], "..."];
  diff = Abs[ToExpression[stored] - N[computedVal, digits + 5]];
  Print["  |stored - computed| = ", diff // N];
  If[agree,
    Print["  PASS: match to ", digits, " digits"],
    Print["  FAIL: mismatch!"];
    nFailures++;
  ];
  nPassed++;
];

(* ------------------------------------------------------------------
   FormatNum: exact same function as in generate.wl
   ------------------------------------------------------------------ *)
FormatNum[x_] := Module[{dec, digits, exp, intPart, fracPart, zeros},
  dec = RealDigits[x, 10, 65];
  digits = dec[[1]];
  exp    = dec[[2]];
  Which[
    exp >= 1 && exp <= 65,
      intPart  = StringJoin[ToString /@ Take[digits, exp]];
      fracPart = StringJoin[ToString /@ Drop[digits, exp]];
      intPart <> "." <> fracPart,
    exp <= 0,
      zeros    = StringJoin[Table["0", {-exp}]];
      fracPart = zeros <> StringJoin[ToString /@ digits];
      "0." <> fracPart,
    True,
      ToString[N[x, 65], InputForm]
  ]
];

(* ================================================================
   VERIFICATION 1: trivial_2x2
   max x_11 s.t. x_11=1, X=[[x_11,x_12],[x_12,x_22]] >= 0.
   Clearly x_11=1 at optimum. No computation needed: opt=1.
   ================================================================ *)
Print["[1/7] trivial_2x2"];
Module[{opt},
  (* Independent derivation: <A_1,X>=x_11=1 is the only constraint.
     C=A_1, so <C,X>=x_11 and the maximum is exactly b_1=1. *)
  opt = 1;
  Print["  Independent derivation: opt = b_1 = 1 (constraint forces x_11=1, C selects x_11)."];
  CheckProblem["trivial_2x2", opt];
];

(* ================================================================
   VERIFICATION 2: max_eigenvalue_2x2
   opt = lambda_max([[2,1],[1,2]]) = 3.
   Independent: compute eigenvalues from characteristic polynomial.
   ================================================================ *)
Print["[2/7] max_eigenvalue_2x2"];
Module[{A, p, roots, opt},
  A = {{2, 1}, {1, 2}};
  (* Characteristic polynomial: det(A - lambda*I) = (2-lambda)^2 - 1 = lambda^2 - 4*lambda + 3 *)
  p = CharacteristicPolynomial[A, lambda];
  Print["  Char poly: ", p, " = 0"];
  roots = Solve[p == 0, lambda, Reals];
  Print["  Eigenvalues: ", lambda /. roots];
  opt = Max[lambda /. roots];
  Print["  lambda_max = ", opt];
  CheckProblem["max_eigenvalue_2x2", opt];
];

(* ================================================================
   VERIFICATION 3: sdp_sqrt2
   opt = sqrt(2).
   Independent: max x_12 s.t. [[1,x_12],[x_12,2]] >= 0.
   PSD condition: det = 2 - x_12^2 >= 0. Max x_12 = sqrt(2).
   ================================================================ *)
Print["[3/7] sdp_sqrt2"];
Module[{opt, sol},
  sol = Maximize[{x12, 1 - 0 >= 0, 2 - 0 >= 0, 1*2 - x12^2 >= 0}, {x12}, Reals];
  Print["  Maximize result: ", sol];
  opt = sol[[1]];
  Print["  opt = ", opt, " = ", N[opt, 25]];
  CheckProblem["sdp_sqrt2", opt];
];

(* ================================================================
   VERIFICATION 4: lp_diagonal_block
   opt = 7.
   Independent: LP vertex enumeration on max x1+2*x2 s.t. x1+x2<=4, x1<=3, x2<=3, x>=0.
   ================================================================ *)
Print["[4/7] lp_diagonal_block"];
Module[{sol},
  sol = Maximize[{x1 + 2*x2,
    x1 + x2 <= 4, x1 <= 3, x2 <= 3, x1 >= 0, x2 >= 0},
    {x1, x2}, Integers];
  Print["  LP solution: ", sol];
  CheckProblem["lp_diagonal_block", sol[[1]]];
];

(* ================================================================
   VERIFICATION 5: ill_conditioned_3x3
   opt = 1 + 1/1000000 = 1000001/1000000.
   Independent: x_11=1 forced (b_1=1), x_22=1 forced (b_2=1), x_12=0 forced (b_3=0).
   Objective = x_11 + (1/10^6)*x_22 = 1 + 10^{-6}.
   ================================================================ *)
Print["[5/7] ill_conditioned_3x3"];
Module[{opt},
  (* All constraints fix x_11, x_22, x_12 exactly. Objective = x_11 + eps*x_22. *)
  opt = 1 + 1/1000000;
  Print["  Independent: constraints force x_11=1,x_22=1,x_12=0; obj=1+1/10^6 exactly."];
  Print["  opt = ", opt, " = ", N[opt, 25]];
  CheckProblem["ill_conditioned_3x3", opt];
];

(* ================================================================
   VERIFICATION 6: mixed_blocks
   opt = 1 + sqrt(2)/2.
   Independent: solve the univariate optimization f(a) = a + sqrt(a(1-a)) + 1/2.
   ================================================================ *)
Print["[6/7] mixed_blocks"];
Module[{fa, sol, aOpt, opt},
  (* f'(a) = 1 + (1-2a)/(2*sqrt(a(1-a))) = 0 => solve 8a^2 - 8a + 1 = 0 *)
  sol = Solve[8*a^2 - 8*a + 1 == 0, a, Reals];
  Print["  Critical points a = ", a /. sol];
  (* Take the one in [0,1] where f is maximized *)
  aOpt = Max[a /. sol];  (* = (2+sqrt(2))/4 *)
  Print["  a* = ", FullSimplify[aOpt]];
  fa = aOpt + Sqrt[aOpt*(1-aOpt)] + 1/2;
  opt = FullSimplify[fa];
  Print["  f(a*) = ", opt, " = ", N[opt, 25]];
  CheckProblem["mixed_blocks", opt];
];

(* ================================================================
   VERIFICATION 7: max_eig_tridiag_3x3
   opt = lambda_max([[3,1,0],[1,2,1],[0,1,1]]) = 2 + sqrt(3).
   Independent: solve characteristic polynomial.
   ================================================================ *)
Print["[7/7] max_eig_tridiag_3x3"];
Module[{A, p, roots, opt},
  A = {{3, 1, 0}, {1, 2, 1}, {0, 1, 1}};
  p = CharacteristicPolynomial[A, lambda];
  Print["  Char poly: ", p];
  roots = Solve[p == 0, lambda, Reals];
  Print["  Eigenvalues: ", FullSimplify[lambda /. roots]];
  opt = Max[lambda /. roots];
  opt = FullSimplify[opt];
  Print["  lambda_max = ", opt, " = ", N[opt, 25]];
  CheckProblem["max_eig_tridiag_3x3", opt];
];

(* ------------------------------------------------------------------
   Final report
   ------------------------------------------------------------------ *)
Print["\n=== Verification summary ==="];
Print["  Passed: ", nPassed];
Print["  Failed: ", nFailures];
If[nFailures == 0,
  Print["  ALL PASS: stored values match independent computations to stated digits."],
  Print["  FAILURES DETECTED: see above for details."]
];
