// tslint:disable:no-console
// import {select, selectParams, simpleShape} from "@syntax/select";
import * as e from "./generated/example";

import {reflection as $} from "edgedb";
import {
  BaseExpression,
  linkDescShape,
  ObjectTypeExpression,
  Poly,
  selectParams,
} from "reflection";

e.str("asdf");
e.bigint(BigInt(1234));

e.$Array(e.$Str);
e.$NamedTuple({asdf: e.$Str});
const asdf = e.$UnnamedTuple([e.$Str]);
e.literal(asdf, ["asdf"]);

e.cast(e.$Str, e.int64(1234));
e.set(e.Hero, e.Villain);

e.select(e.Hero, {
  villains: {nemesis: {villains: {nemesis: true}}},
});

// e.$Hero.__shape__.__type__.name;
// e.default.Hero;

const query = e.select(
  e.Person,
  {
    id: true,
    __type__: {
      id: true,
      name: true,
    },
    name: 1 > 0,
    computed: e.str("person"),
  },
  e.shape(e.Hero, {secret_identity: true, __type__: {name: true}}),
  e.shape(e.Villain, {
    nemesis: {id: true},
    name: true,
  })
).__element__.__tstype__;

const q2 = e.select(e.Villain.nemesis, {
  number_of_movies: true,
  secret_identity: true,
  id: true,
});

type params = selectParams<typeof q2>;
const asfdf = e.select(q2, {
  name: true,
  id: true,
  number_of_movies: true,
}).__element__.__tstype__;

const qwers = e.select(e.Movie, {
  title: true,
  characters: {
    name: true,
    "@ac": true,
  },
}).__element__.__tstype__;
