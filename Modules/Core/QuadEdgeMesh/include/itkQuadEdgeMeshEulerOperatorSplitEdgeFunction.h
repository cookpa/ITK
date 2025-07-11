/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkQuadEdgeMeshEulerOperatorSplitEdgeFunction_h
#define itkQuadEdgeMeshEulerOperatorSplitEdgeFunction_h

#include "itkQuadEdgeMeshEulerOperatorSplitVertexFunction.h"

namespace itk
{
/**
 * \class QuadEdgeMeshEulerOperatorSplitEdgeFunction
 * \brief Given Edge is split into two and associated faces see their
 * degree increased by one (a triangle is transformed into a quad for
 * example).
 *
 * \ingroup QEMeshModifierFunctions
 * \ingroup ITKQuadEdgeMesh
 */
template <typename TMesh, typename TQEType>
class ITK_TEMPLATE_EXPORT QuadEdgeMeshEulerOperatorSplitEdgeFunction : public QuadEdgeMeshFunctionBase<TMesh, TQEType *>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(QuadEdgeMeshEulerOperatorSplitEdgeFunction);

  /** Standard class type aliases. */
  using Self = QuadEdgeMeshEulerOperatorSplitEdgeFunction;
  using Superclass = QuadEdgeMeshFunctionBase<TMesh, TQEType *>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  itkNewMacro(Self);
  /** \see LightObject::GetNameOfClass() */
  itkOverrideGetNameOfClassMacro(QuadEdgeMeshEulerOperatorSplitEdgeFunction);

  /** Type of QuadEdge with which to apply slicing. */
  using QEType = TQEType;

  using typename Superclass::MeshType;
  using typename Superclass::OutputType;
  using PointIdentifier = typename MeshType::PointIdentifier;

  using SplitVertex = QuadEdgeMeshEulerOperatorSplitVertexFunction<MeshType, QEType>;

  /** Evaluate at the specified input position */
  virtual OutputType
  Evaluate(QEType * e)
  {
    if (!e)
    {
      itkDebugMacro("Input is not an edge.");
      return ((QEType *)nullptr);
    }

    if (!this->m_Mesh)
    {
      itkDebugMacro("No mesh present.");
      return ((QEType *)nullptr);
    }

    m_SplitVertex->SetInput(this->m_Mesh);
    return (m_SplitVertex->Evaluate(e->GetLprev(), e->GetSym()));
  }

  const PointIdentifier
  GetNewPointID()
  {
    return (m_SplitVertex->GetNewPointID());
  }

protected:
  QuadEdgeMeshEulerOperatorSplitEdgeFunction()
    : m_SplitVertex(SplitVertex::New())
  {}

  ~QuadEdgeMeshEulerOperatorSplitEdgeFunction() override = default;

private:
  typename SplitVertex::Pointer m_SplitVertex{};
};
} // end namespace itk

#endif // itkQuadEdgeMeshEulerOperatorSplitEdgeFunction.h
